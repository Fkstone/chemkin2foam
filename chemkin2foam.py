import os
import re
import math
import shutil
import subprocess

import cantera as ct
import numpy as np
from scipy.optimize import curve_fit


# ============================================================
# Basic utilities
# ============================================================

R_CAL = 1.987


def format_value(val):
    return f"{val:.9g}"


# ============================================================
# Interactive user input
# ============================================================

def ask_user_settings():
    print("==============================================")
    print(" Chemkin to OpenFOAM 8-13 Mechanism Converter ")
    print("==============================================")
    print()

    ck_input_dir = input("Chemkin input directory: ").strip()
    ck_mech_file = input("Chemkin mechanism file name: ").strip()
    ck_thermo_file = input("Chemkin thermo file name: ").strip()
    ck_transport_file = input("Chemkin transport file name: ").strip()
    output_dir = input("OpenFOAM output directory: ").strip()

    if not ck_input_dir:
        ck_input_dir = "."

    if not output_dir:
        output_dir = "./chemkin"

    permissive_input = input("Use ck2yaml --permissive? [y/N]: ").strip().lower()
    permissive = permissive_input in ["y", "yes"]

    pressure_input = input(
        "Pressure for PLOG reactions in atm, empty = keep PLOG format: "
    ).strip()

    if pressure_input:
        pressure_atm = float(pressure_input)
    else:
        pressure_atm = None

    print()
    print("Input summary:")
    print(f"  Chemkin input directory : {ck_input_dir}")
    print(f"  Mechanism file          : {ck_mech_file}")
    print(f"  Thermo file             : {ck_thermo_file}")
    print(f"  Transport file          : {ck_transport_file}")
    print(f"  Output directory        : {output_dir}")
    print(f"  ck2yaml permissive      : {permissive}")
    print(f"  PLOG pressure           : {pressure_atm}")
    print()

    confirm = input("Start conversion? [Y/n]: ").strip().lower()

    if confirm in ["n", "no"]:
        print("Conversion cancelled.")
        exit(0)

    return (
        ck_input_dir,
        ck_mech_file,
        ck_thermo_file,
        ck_transport_file,
        output_dir,
        permissive,
        pressure_atm,
    )


# ============================================================
# Chemkin to Cantera YAML
# ============================================================

def chemkin_to_yaml(ck_mech, ck_thermo, ck_transport, yaml_path, permissive=False):
    """
    Convert Chemkin-format mechanism files to Cantera YAML using ck2yaml.
    """

    if shutil.which("ck2yaml") is not None:
        cmd = [
            "ck2yaml",
            f"--input={ck_mech}",
            f"--thermo={ck_thermo}",
            f"--transport={ck_transport}",
            f"--output={yaml_path}",
        ]
    else:
        cmd = [
            "python",
            "-m",
            "cantera.ck2yaml",
            f"--input={ck_mech}",
            f"--thermo={ck_thermo}",
            f"--transport={ck_transport}",
            f"--output={yaml_path}",
        ]

    if permissive:
        cmd.append("--permissive")

    print()
    print("Running ck2yaml:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"ck2yaml failed to generate: {yaml_path}")

    print(f"Generated Cantera YAML: {yaml_path}")


# ============================================================
# Sutherland transport fitting
# ============================================================

def sutherland(x, As, Ts):
    return As * x**(3 / 2) / (Ts + x)


def transport_header():
    head = '/*--------------------------------*- C++ -*----------------------------------*\\ \n  =========                 |\n  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n   \\\\    /   O peration     | Website:  https://openfoam.org\n    \\\\  /    A nd           | Version:  13\n     \\\\/     M anipulation  |\n\\*---------------------------------------------------------------------------*/\nFoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    location    "chemkin";\n    object      transportProperties;\n}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
    return head


def generate_transport_from_cantera(gas, output_dir=None):
    """
    Generate Sutherland As/Ts from Cantera pure-species viscosity.
    """

    default = (1.0e-6, 120.0)
    table = {}

    Temp = np.arange(200, 2501, 1)
    texts = [transport_header()]

    T0 = gas.T
    P0 = gas.P
    X0 = gas.X

    for species in gas.species_names:
        print("Species = ", species)
        print("Species_Index = ", gas.species_index(species))

        try:
            gas.X = f"{species}:1"

            muy = []

            for temp in Temp:
                gas.TP = temp, ct.one_atm
                muy.append(gas.viscosity)

            popt, pcov = curve_fit(
                sutherland,
                Temp,
                muy,
                p0=[1.0e-6, 120.0],
                maxfev=10000
            )

            As = float(popt[0])
            Ts = float(popt[1])

        except Exception as e:
            print(f"Warning: failed to fit transport for {species}: {e}")
            print("Using default As/Ts.")
            As, Ts = default

        print("As = ", As, "\n", "Ts = ", Ts)
        print("#######################################")

        table[species] = (As, Ts)

        string = '"{0}"\n{1}\n    transport\n    {2}\n        As {3};\n        Ts {4};\n    {5}\n{6}\n'.format(
            species,
            "{",
            "{",
            format_value(As),
            format_value(Ts),
            "}",
            "}"
        )

        texts.append(string)

    texts.append("// ************************************************************************* //")

    gas.TPX = T0, P0, X0

    if output_dir is not None:
        with open(os.path.join(output_dir, "transportProperties"), "w") as f:
            f.write("\n".join(texts) + "\n")

    return default, table


# ============================================================
# Converter logic for OpenFOAM 8-13
# ============================================================

def extract_true_third_body_contents(s: str):
    pattern = r"\+\s*(M)\b"

    returnValue = re.findall(pattern, s)

    if returnValue:
        return returnValue[0]
    else:
        return None


def extract_third_body_content(s: str):
    pattern = r"\(\s*\+([^\)\s]+)\s*\)"

    returnValue = re.findall(pattern, s)

    if returnValue:
        return returnValue[0]
    else:
        return None


def species_block(spec, default, table):
    thermo = spec.thermo
    coeffs = list(thermo.coeffs)

    Tcommon = coeffs[0]
    high = coeffs[1:8]
    low = coeffs[8:]

    if (
        math.isclose(Tcommon, thermo.max_temp)
        and all(math.isclose(h, l, rel_tol=0, abs_tol=1e-12) for h, l in zip(high, low))
    ):
        Tcommon = 1000.0

    As, Ts = table.get(spec.name, default)

    lines = []

    lines.append(f"{spec.name}")
    lines.append("{")

    lines.append("    specie")
    lines.append("    {")
    lines.append(f"        molWeight       {format_value(spec.molecular_weight)};")
    lines.append("    }")

    lines.append("    thermodynamics")
    lines.append("    {")
    lines.append(f"        Tlow            {format_value(thermo.min_temp)};")
    lines.append(f"        Thigh           {format_value(thermo.max_temp)};")
    lines.append(f"        Tcommon         {format_value(Tcommon)};")
    lines.append("        highCpCoeffs    ( " + " ".join(format_value(v) for v in high) + " );")
    lines.append("        lowCpCoeffs     ( " + " ".join(format_value(v) for v in low) + " );")
    lines.append("    }")

    lines.append("    transport")
    lines.append("    {")
    lines.append(f"        As              {format_value(As)};")
    lines.append(f"        Ts              {format_value(Ts)};")
    lines.append("    }")

    lines.append("    elements")
    lines.append("    {")
    for el, amt in spec.composition.items():
        lines.append(f"        {el}               {format_value(amt)};")
    lines.append("    }")

    lines.append("}")

    return "\n".join(lines)


def writeSpecies(gas):
    species_names = gas.species_names

    lines = [
        "species         %d ( %s );" % (gas.n_species, " ".join(species_names)),
        "",
    ]

    return lines, species_names


def elements_block_lines(gas):
    element_names = gas.element_names

    lines = ["elements", str(len(element_names)), "("]
    lines += element_names
    lines += [")", ";"]

    return lines


def writeThermo(gas, default_tp, table_tp, output_dir, species_names, header_lines=None):
    thermo_lines = list(header_lines or [])

    for sp_name in species_names:
        sp = gas.species(sp_name)
        thermo_lines.append(species_block(sp, default_tp, table_tp))
        thermo_lines.append("")

    with open(os.path.join(output_dir, "thermos"), "w") as f:
        f.write("\n".join(thermo_lines) + "\n")


def writeReactions(
    gas,
    species_names,
    output_dir,
    pressure_atm=None,
    header_lines=None,
    elements_lines=None,
    rtype_suffix="",
    thirdBodyCase="T"
):
    rxn_lines = []

    if header_lines:
        rxn_lines.extend(header_lines)

    if elements_lines:
        rxn_lines.extend(elements_lines)
        rxn_lines.append("")

    rxn_lines += ["reactions", "{"]

    rxns = gas.reactions()

    i = 0
    out_idx = 0

    while i < len(rxns):
        rxn = rxns[i]

        try:
            if i + 1 < len(rxns) and is_reverse_pair(rxn, rxns[i + 1]):
                rxn_lines.append(
                    combined_reaction_block(
                        rxn,
                        rxns[i + 1],
                        out_idx,
                        species_names,
                        rtype_suffix=rtype_suffix,
                        thirdBodyCase=thirdBodyCase
                    )
                )
                i += 2

            else:
                rxn_lines.append(
                    reaction_block(
                        rxn,
                        out_idx,
                        species_names,
                        pressure_pa=pressure_atm * ct.one_atm if pressure_atm else None,
                        rtype_suffix=rtype_suffix,
                        thirdBodyCase=thirdBodyCase
                    )
                )
                i += 1

            out_idx += 1

        except NotImplementedError as e:
            print(f"Skipping reaction {i} ({rxn.equation}): {e}")
            i += 1

    rxn_lines.append("}")
    rxn_lines.append("Tlow            250;")
    rxn_lines.append("Thigh           5000;")

    with open(os.path.join(output_dir, "reactions"), "w") as f:
        f.write("\n".join(rxn_lines) + "\n")


def arrhenius_params(rate):
    A = rate.pre_exponential_factor
    beta = rate.temperature_exponent
    Ta = rate.activation_energy / ct.gas_constant

    return A, beta, Ta


def arrhenius_at_pressure(rates, pressure_pa):
    rates = sorted(rates, key=lambda pr: pr[0])

    for p, rate in rates:
        if abs(p - pressure_pa) / pressure_pa < 1e-6:
            return arrhenius_params(rate)

    for (p_i, rate_i), (p_j, rate_j) in zip(rates[:-1], rates[1:]):
        if p_i <= pressure_pa <= p_j:
            A_i, b_i, Ta_i = arrhenius_params(rate_i)
            A_j, b_j, Ta_j = arrhenius_params(rate_j)

            k = (math.log(pressure_pa) - math.log(p_i)) / (
                math.log(p_j) - math.log(p_i)
            )

            A = A_i * (A_j / A_i) ** k
            b = b_i + k * (b_j - b_i)
            Ta = Ta_i + k * (Ta_j - Ta_i)

            return A, b, Ta

    raise ValueError(
        "Pressure %.3g Pa outside range of provided PLOG data" % pressure_pa
    )


def format_equation(rxn):
    eq = rxn.equation.replace("<=>", "=").replace("=>", "=")
    eq = re.sub(r"\s*\(\+\s*[A-Za-z0-9_]+\s*\)", "", eq)
    eq = re.sub(r"\s*\+ M", "", eq)
    eq = re.sub(r"(\d+)\s+(\w)", r"\1\2", eq)
    eq = re.sub(r"\s+", " ", eq).strip()

    orders = getattr(rxn, "orders", {})

    if orders:
        pattern = re.compile(r"(\b\d*\.?\d*)([A-Za-z0-9_]+)")
        left, right = [s.strip() for s in eq.split("=")]

        def repl(match):
            coeff_str = match.group(1)
            species = match.group(2)
            stoich = float(coeff_str) if coeff_str else 1.0
            order = orders.get(species)

            if order is not None and abs(order - stoich) > 1e-9:
                return f"{coeff_str}{species}^{format_value(order)}"

            return f"{coeff_str}{species}"

        left = pattern.sub(repl, left)
        right = pattern.sub(repl, right)
        eq = f"{left} = {right}"

    return eq


def third_body_block(
    efficiencies,
    species_names,
    pure_third_body=True,
    must_return=False,
    only_one_third_body=False
):
    if not efficiencies and not must_return:
        return []

    if not efficiencies:
        efficiencies = {sp: 1.0 for sp in species_names}

    default_value = 0.0 if only_one_third_body else 1.0

    lines = []

    if not pure_third_body:
        lines += [
            "        thirdBodyEfficiencies",
            "        {"
        ]

    lines += [
        "            coeffs",
        "                " + str(len(species_names)),
        "            (",
    ]

    for sp in species_names:
        val = efficiencies.get(sp, default_value)
        lines.append(f"            ({sp} {format_value(val)})")

    lines += [
        "            )",
        "            ;",
    ]

    if not pure_third_body:
        lines.append("        }")

    return lines


def k_block(name, rate):
    A, beta, Ta = arrhenius_params(rate)

    return [
        f"        {name}",
        "        {",
        f"            A               {format_value(A)};",
        f"            beta            {format_value(beta)};",
        f"            Ta              {format_value(Ta)};",
        "        }",
    ]


def k0_block(rate):
    return k_block("k0", rate)


def kinf_block(rate):
    return k_block("kInf", rate)


def F_block(alpha=None, T3=None, T1=None, T2=None):
    lines = ["        F", "        {"]

    if alpha is not None:
        lines.append(f"            alpha           {format_value(alpha)};")

    if T3 is not None:
        lines.append(f"            Tsss            {format_value(T3)};")

    if T1 is not None:
        lines.append(f"            Ts              {format_value(T1)};")

    if T2 is not None:
        lines.append(f"            Tss             {format_value(T2)};")

    lines.append("        }")

    return lines


def plog_block(rates):
    lines = ["        ArrheniusData", "        ("]

    for p, rate in rates:
        A, beta, Ta = arrhenius_params(rate)
        lines.append(
            f"            ({format_value(p)}  {format_value(A)} {format_value(beta)} {format_value(Ta)})"
        )

    lines += ["        )", "        ;"]

    return lines


def base_block(rtype, rxn, rate):
    A, beta, Ta = arrhenius_params(rate)

    return [
        f"        type            {rtype};",
        f"        reaction        \"{format_equation(rxn)}\";",
        f"        A               {format_value(A)};",
        f"        beta            {format_value(beta)};",
        f"        Ta              {format_value(Ta)};",
    ]


def reaction_block(
    rxn,
    index,
    species_names,
    pressure_pa=None,
    rtype_suffix="",
    thirdBodyCase="T"
):
    only_one_third_body = is_one_third_body(rxn)

    prefix = "reversible" if rxn.reversible else "irreversible"

    if rxn.reaction_type == "Arrhenius":
        rtype = f"{prefix}Arrhenius{rtype_suffix}"
        body = base_block(rtype, rxn, rxn.rate)

    elif rxn.reaction_type == "three-body-Arrhenius":
        if rxn.input_data.get("type") == "three-body":
            rtype = f"{prefix}{thirdBodyCase}hirdBodyArrhenius{rtype_suffix}"
            body = base_block(rtype, rxn, rxn.rate)
            body += third_body_block(
                rxn.third_body.efficiencies,
                species_names,
                must_return=not rxn.third_body.efficiencies,
                only_one_third_body=only_one_third_body,
            )
        else:
            rtype = f"{prefix}Arrhenius{rtype_suffix}"
            body = base_block(rtype, rxn, rxn.rate)

    elif rxn.reaction_type == "falloff-Troe":
        rtype = f"{prefix}ArrheniusTroeFallOff{rtype_suffix}"

        body = [
            f"        type            {rtype};",
            f"        reaction        \"{format_equation(rxn)}\";",
        ]

        body += k0_block(rxn.rate.low_rate)
        body += kinf_block(rxn.rate.high_rate)

        if len(rxn.rate.falloff_coeffs) == 3:
            alpha, T3, T1 = rxn.rate.falloff_coeffs
            body += F_block(alpha, T3, T1, 4.5036e15)
        else:
            alpha, T3, T1, T2 = rxn.rate.falloff_coeffs
            body += F_block(alpha, T3, T1, T2)

        body += third_body_block(
            rxn.third_body.efficiencies,
            species_names,
            pure_third_body=False,
            must_return=not rxn.third_body.efficiencies,
            only_one_third_body=only_one_third_body,
        )

    elif rxn.reaction_type == "falloff-Lindemann":
        rtype = f"{prefix}ArrheniusLindemannFallOff{rtype_suffix}"

        body = [
            f"        type            {rtype};",
            f"        reaction        \"{format_equation(rxn)}\";",
        ]

        body += k0_block(rxn.rate.low_rate)
        body += kinf_block(rxn.rate.high_rate)
        body += F_block()

        body += third_body_block(
            rxn.third_body.efficiencies,
            species_names,
            pure_third_body=False,
            must_return=not rxn.third_body.efficiencies,
            only_one_third_body=only_one_third_body,
        )

    elif rxn.reaction_type == "pressure-dependent-Arrhenius":
        if pressure_pa is None:
            rtype = f"{prefix}ArrheniusPLOG{rtype_suffix}"
            first_p, first_rate = rxn.rate.rates[0]
            body = base_block(rtype, rxn, first_rate)
            body += plog_block(rxn.rate.rates)

        else:
            rtype = f"{prefix}Arrhenius{rtype_suffix}"

            try:
                A, beta, Ta = arrhenius_at_pressure(rxn.rate.rates, pressure_pa)
                rate = ct.Arrhenius(A, beta, Ta * ct.gas_constant)

            except ValueError as exc:
                print(
                    f"Warning: {exc}. Using nearest tabulated pressure for {rxn.equation}."
                )

                rates = sorted(
                    rxn.rate.rates,
                    key=lambda pr: abs(pr[0] - pressure_pa)
                )

                rate = rates[0][1]

            body = base_block(rtype, rxn, rate)

    else:
        raise NotImplementedError(f"Reaction type {rxn.reaction_type} not supported")

    lines = [f"    un-named-reaction-{index}", "    {"]
    lines.extend(body)
    lines.append("    }")

    return "\n".join(lines)


def is_reverse_pair(r1, r2):
    if r1.reaction_type != r2.reaction_type:
        return False

    if r1.reversible or r2.reversible:
        return False

    def same_stoich(a, b):
        if set(a.keys()) != set(b.keys()):
            return False

        for k in a:
            if abs(a[k] - b[k]) > 1e-12:
                return False

        return True

    return same_stoich(r1.reactants, r2.products) and same_stoich(r1.products, r2.reactants)


def combined_reaction_block(
    forward,
    reverse,
    index,
    species_names,
    rtype_suffix="",
    thirdBodyCase="T"
):
    if forward.reaction_type == "Arrhenius":
        rtype = f"nonEquilibriumReversibleArrhenius{rtype_suffix}"

    elif forward.reaction_type == "three-body-Arrhenius":
        rtype = f"nonEquilibriumReversible{thirdBodyCase}hirdBodyArrhenius{rtype_suffix}"

    else:
        raise NotImplementedError(
            f"Non-equilibrium reversible type for {forward.reaction_type} not supported"
        )

    lines = [f"    un-named-reaction-{index}", "    {"]
    lines.append(f"        type            {rtype};")
    lines.append(f"        reaction        \"{format_equation(forward)}\";")

    lines.append("        forward")
    lines.append("        {")

    A, beta, Ta = arrhenius_params(forward.rate)
    lines.append(f"            A               {format_value(A)};")
    lines.append(f"            beta            {format_value(beta)};")
    lines.append(f"            Ta              {format_value(Ta)};")

    if forward.reaction_type == "three-body-Arrhenius":
        lines += third_body_block(
            forward.third_body.efficiencies,
            species_names,
            must_return=not forward.third_body.efficiencies,
        )

    lines.append("        }")

    lines.append("        reverse")
    lines.append("        {")

    A, beta, Ta = arrhenius_params(reverse.rate)
    lines.append(f"            A               {format_value(A)};")
    lines.append(f"            beta            {format_value(beta)};")
    lines.append(f"            Ta              {format_value(Ta)};")

    if reverse.reaction_type == "three-body-Arrhenius":
        lines += third_body_block(
            reverse.third_body.efficiencies,
            species_names,
            must_return=not reverse.third_body.efficiencies,
        )

    lines.append("        }")
    lines.append("    }")

    return "\n".join(lines)


def is_one_third_body(rxn):
    isLindemann = rxn.reaction_type == "falloff-Lindemann"
    isTroe = rxn.reaction_type == "falloff-Troe"
    isThirdBody = rxn.reaction_type == "three-body-Arrhenius"

    is_not_one_third_body = extract_true_third_body_contents(rxn.equation) == "M"

    if not (isLindemann or isTroe or isThirdBody):
        return False

    if is_not_one_third_body:
        return False

    if not rxn.reversible:
        return False

    third_body_species = extract_third_body_content(rxn.equation)

    if third_body_species == "M":
        return False

    return True


def default_converter(gas, default_tp, table_tp, output_dir, pressure_atm=None):
    header_lines, species_names = writeSpecies(gas)

    writeThermo(
        gas,
        default_tp,
        table_tp,
        output_dir,
        species_names,
        header_lines
    )

    with open(os.path.join(output_dir, "elements"), "w") as f:
        f.write("\n".join(elements_block_lines(gas)) + "\n")

    writeReactions(
        gas,
        species_names,
        output_dir,
        pressure_atm,
        rtype_suffix="",
    )


# ============================================================
# Main conversion function
# ============================================================

def convert_from_chemkin_to_openfoam(
    ck_input_dir,
    ck_mech_file,
    ck_thermo_file,
    ck_transport_file,
    output_dir,
    permissive=False,
    pressure_atm=None,
):
    os.makedirs(output_dir, exist_ok=True)

    ck_mech = os.path.join(ck_input_dir, ck_mech_file)
    ck_thermo = os.path.join(ck_input_dir, ck_thermo_file)
    ck_transport = os.path.join(ck_input_dir, ck_transport_file)

    yaml_path = os.path.join(output_dir, "chem.yaml")

    if not os.path.exists(ck_mech):
        raise FileNotFoundError(f"Chemkin mechanism file not found: {ck_mech}")

    if not os.path.exists(ck_thermo):
        raise FileNotFoundError(f"Chemkin thermo file not found: {ck_thermo}")

    if not os.path.exists(ck_transport):
        raise FileNotFoundError(f"Chemkin transport file not found: {ck_transport}")

    chemkin_to_yaml(
        ck_mech,
        ck_thermo,
        ck_transport,
        yaml_path,
        permissive=permissive
    )

    gas = ct.Solution(yaml_path)

    print()
    print("Generating Sutherland As/Ts from Cantera transport data...")
    print()

    default_tp, table_tp = generate_transport_from_cantera(
        gas,
        output_dir=output_dir
    )

    default_converter(
        gas,
        default_tp,
        table_tp,
        output_dir,
        pressure_atm
    )

    print()
    print("Conversion finished.")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print(f"  {os.path.join(output_dir, 'chem.yaml')}")
    print(f"  {os.path.join(output_dir, 'thermos')}")
    print(f"  {os.path.join(output_dir, 'reactions')}")
    print(f"  {os.path.join(output_dir, 'elements')}")
    print(f"  {os.path.join(output_dir, 'transportProperties')}")


if __name__ == "__main__":
    (
        ck_input_dir,
        ck_mech_file,
        ck_thermo_file,
        ck_transport_file,
        output_dir,
        permissive,
        pressure_atm,
    ) = ask_user_settings()

    convert_from_chemkin_to_openfoam(
        ck_input_dir=ck_input_dir,
        ck_mech_file=ck_mech_file,
        ck_thermo_file=ck_thermo_file,
        ck_transport_file=ck_transport_file,
        output_dir=output_dir,
        permissive=permissive,
        pressure_atm=pressure_atm,
    )