import os
import re
import math
import shutil
import subprocess

import cantera as ct
import numpy as np
from scipy.optimize import curve_fit

R_CAL = 1.987
fmt = lambda v: f"{v:.9g}"


# ── User input ────────────────────────────────────────────────────────────────

def ask_user_settings():
    print("==============================================")
    print(" Chemkin to OpenFOAM 8-13 Mechanism Converter")
    print("==============================================\n")

    fields = [
        ("Chemkin input directory",     "."),
        ("Chemkin mechanism file name",  ""),
        ("Chemkin thermo file name",     ""),
        ("Chemkin transport file name",  ""),
        ("OpenFOAM output directory",    "./chemkin"),
    ]
    vals = []
    for prompt, default in fields:
        v = input(f"{prompt}: ").strip()
        vals.append(v or default)

    ck_input_dir, ck_mech_file, ck_thermo_file, ck_transport_file, output_dir = vals

    permissive = input("Use ck2yaml --permissive? [y/N]: ").strip().lower() in ("y", "yes")
    p_str = input("Pressure for PLOG reactions in atm, empty = keep PLOG format: ").strip()
    pressure_atm = float(p_str) if p_str else None

    print(f"\nInput summary:")
    for k, v in zip(
        ["Chemkin dir", "Mech", "Thermo", "Transport", "Output", "Permissive", "PLOG pressure"],
        [ck_input_dir, ck_mech_file, ck_thermo_file, ck_transport_file,
         output_dir, permissive, pressure_atm]
    ):
        print(f"  {k:<20}: {v}")

    if input("\nStart conversion? [Y/n]: ").strip().lower() in ("n", "no"):
        print("Conversion cancelled.")
        exit(0)

    return ck_input_dir, ck_mech_file, ck_thermo_file, ck_transport_file, output_dir, permissive, pressure_atm


# ── Chemkin → Cantera YAML ────────────────────────────────────────────────────

def chemkin_to_yaml(ck_mech, ck_thermo, ck_transport, yaml_path, permissive=False):
    base = "ck2yaml" if shutil.which("ck2yaml") else "python -m cantera.ck2yaml"
    cmd = (base.split() +
           [f"--input={ck_mech}", f"--thermo={ck_thermo}",
            f"--transport={ck_transport}", f"--output={yaml_path}"]
           + (["--permissive"] if permissive else []))
    print("\nRunning:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"ck2yaml failed to generate: {yaml_path}")


# ── Sutherland transport fitting ──────────────────────────────────────────────

_FOAM_HEADER = '''\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  13
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "chemkin";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
'''

def _sutherland(T, As, Ts):
    return As * T ** 1.5 / (Ts + T)


def generate_transport_from_cantera(gas, output_dir=None):
    Temp = np.arange(200, 2501, 1)
    DEFAULT = (1.0e-6, 120.0)
    table = {}
    texts = [_FOAM_HEADER]

    T0, P0, X0 = gas.T, gas.P, gas.X.copy()

    for sp in gas.species_names:
        try:
            gas.X = f"{sp}:1"
            muy = [gas.viscosity for gas.TP in ((t, ct.one_atm) for t in Temp)]
            (As, Ts), _ = curve_fit(_sutherland, Temp, muy, p0=DEFAULT, maxfev=10000)
        except Exception as e:
            print(f"Warning: failed to fit {sp}: {e}. Using defaults.")
            As, Ts = DEFAULT
        table[sp] = (float(As), float(Ts))
        texts.append(
            f'"{sp}"\n{{\n    transport\n    {{\n'
            f'        As {fmt(As)};\n        Ts {fmt(Ts)};\n    }}\n}}\n'
        )

    texts.append("// ************************************************************************* //")
    gas.TPX = T0, P0, X0

    if output_dir:
        with open(os.path.join(output_dir, "transportProperties"), "w") as f:
            f.write("\n".join(texts) + "\n")

    return DEFAULT, table


# ── Thermo / species block ────────────────────────────────────────────────────

def species_block(spec, default, table):
    th = spec.thermo
    c = list(th.coeffs)
    Tcom = c[0]
    high, low = c[1:8], c[8:]

    if (math.isclose(Tcom, th.max_temp) and
            all(math.isclose(h, l, abs_tol=1e-12) for h, l in zip(high, low))):
        Tcom = 1000.0

    As, Ts = table.get(spec.name, default)
    elems = "\n".join(f"        {el}               {fmt(v)};" for el, v in spec.composition.items())

    return (
        f"{spec.name}\n{{\n"
        f"    specie\n    {{\n        molWeight       {fmt(spec.molecular_weight)};\n    }}\n"
        f"    thermodynamics\n    {{\n"
        f"        Tlow            {fmt(th.min_temp)};\n"
        f"        Thigh           {fmt(th.max_temp)};\n"
        f"        Tcommon         {fmt(Tcom)};\n"
        f"        highCpCoeffs    ( {' '.join(fmt(v) for v in high)} );\n"
        f"        lowCpCoeffs     ( {' '.join(fmt(v) for v in low)} );\n"
        f"    }}\n"
        f"    transport\n    {{\n        As              {fmt(As)};\n        Ts              {fmt(Ts)};\n    }}\n"
        f"    elements\n    {{\n{elems}\n    }}\n}}"
    )


# ── Writers ───────────────────────────────────────────────────────────────────

def writeSpecies(gas):
    lines = [f"species         {gas.n_species} ( {' '.join(gas.species_names)} );", ""]
    return lines, gas.species_names


def writeThermo(gas, default_tp, table_tp, output_dir, species_names, header_lines=None):
    lines = list(header_lines or [])
    for name in species_names:
        lines += [species_block(gas.species(name), default_tp, table_tp), ""]
    with open(os.path.join(output_dir, "thermos"), "w") as f:
        f.write("\n".join(lines) + "\n")


def writeReactions(gas, species_names, output_dir, pressure_atm=None,
                   header_lines=None, elements_lines=None,
                   rtype_suffix="", thirdBodyCase="T"):
    lines = list(header_lines or [])
    if elements_lines:
        lines += elements_lines + [""]
    lines += ["reactions", "{"]

    rxns, i, idx = gas.reactions(), 0, 0
    while i < len(rxns):
        try:
            if i + 1 < len(rxns) and _is_reverse_pair(rxns[i], rxns[i + 1]):
                lines.append(combined_reaction_block(rxns[i], rxns[i+1], idx, species_names,
                                                     rtype_suffix, thirdBodyCase))
                i += 2
            else:
                lines.append(reaction_block(rxns[i], idx, species_names,
                                            pressure_pa=pressure_atm * ct.one_atm if pressure_atm else None,
                                            rtype_suffix=rtype_suffix,
                                            thirdBodyCase=thirdBodyCase))
                i += 1
            idx += 1
        except NotImplementedError as e:
            print(f"Skipping reaction {i} ({rxns[i].equation}): {e}")
            i += 1

    lines += ["}", "Tlow            250;", "Thigh           5000;"]
    with open(os.path.join(output_dir, "reactions"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Reaction helpers ──────────────────────────────────────────────────────────

def _ap(rate):
    return rate.pre_exponential_factor, rate.temperature_exponent, rate.activation_energy / ct.gas_constant


def _arrhenius_at_pressure(rates, pressure_pa):
    rates = sorted(rates, key=lambda pr: pr[0])
    for p, r in rates:
        if abs(p - pressure_pa) / pressure_pa < 1e-6:
            return _ap(r)
    for (pi, ri), (pj, rj) in zip(rates[:-1], rates[1:]):
        if pi <= pressure_pa <= pj:
            k = (math.log(pressure_pa) - math.log(pi)) / (math.log(pj) - math.log(pi))
            Ai, bi, Tai = _ap(ri)
            Aj, bj, Taj = _ap(rj)
            return Ai * (Aj/Ai)**k, bi + k*(bj-bi), Tai + k*(Taj-Tai)
    raise ValueError(f"Pressure {pressure_pa:.3g} Pa outside PLOG range")


def _fmt_eq(rxn):
    eq = rxn.equation.replace("<=>", "=").replace("=>", "=")
    eq = re.sub(r"\s*\(\+\s*[A-Za-z0-9_]+\s*\)", "", eq)
    eq = re.sub(r"\s*\+ M", "", eq)
    eq = re.sub(r"(\d+)\s+(\w)", r"\1\2", eq)
    eq = re.sub(r"\s+", " ", eq).strip()

    orders = getattr(rxn, "orders", {})
    if orders:
        pat = re.compile(r"(\b\d*\.?\d*)([A-Za-z0-9_]+)")
        left, right = (s.strip() for s in eq.split("="))
        def repl(m):
            coeff, sp = m.group(1), m.group(2)
            stoich = float(coeff) if coeff else 1.0
            o = orders.get(sp)
            return f"{coeff}{sp}^{fmt(o)}" if o is not None and abs(o - stoich) > 1e-9 else f"{coeff}{sp}"
        eq = f"{pat.sub(repl, left)} = {pat.sub(repl, right)}"
    return eq


def _third_body_block(efficiencies, species_names, pure=True, must_return=False, only_one=False):
    if not efficiencies and not must_return:
        return []
    default_val = 0.0 if only_one else 1.0
    effs = efficiencies or {sp: 1.0 for sp in species_names}
    inner = (["        thirdBodyEfficiencies", "        {"] if not pure else [])
    inner += ["            coeffs", f"                {len(species_names)}", "            ("]
    inner += [f"            ({sp} {fmt(effs.get(sp, default_val))})" for sp in species_names]
    inner += ["            )", "            ;"]
    if not pure:
        inner.append("        }")
    return inner


def _k_block(name, rate):
    A, beta, Ta = _ap(rate)
    return [f"        {name}", "        {",
            f"            A               {fmt(A)};",
            f"            beta            {fmt(beta)};",
            f"            Ta              {fmt(Ta)};", "        }"]


def _F_block(alpha=None, T3=None, T1=None, T2=None):
    keys = {"alpha": alpha, "Tsss": T3, "Ts": T1, "Tss": T2}
    lines = ["        F", "        {"]
    lines += [f"            {k}           {fmt(v)};" for k, v in keys.items() if v is not None]
    lines.append("        }")
    return lines


def _base_block(rtype, rxn, rate):
    A, beta, Ta = _ap(rate)
    return [f"        type            {rtype};",
            f"        reaction        \"{_fmt_eq(rxn)}\";",
            f"        A               {fmt(A)};",
            f"        beta            {fmt(beta)};",
            f"        Ta              {fmt(Ta)};"]


def _is_one_third_body(rxn):
    if rxn.reaction_type not in ("falloff-Lindemann", "falloff-Troe", "three-body-Arrhenius"):
        return False
    if re.search(r"\+\s*M\b", rxn.equation):
        return False
    if not rxn.reversible:
        return False
    tb = re.findall(r"\(\s*\+([^\)\s]+)\s*\)", rxn.equation)
    return not (tb and tb[0] == "M")


def _is_reverse_pair(r1, r2):
    if r1.reaction_type != r2.reaction_type or r1.reversible or r2.reversible:
        return False
    def same(a, b):
        return set(a) == set(b) and all(abs(a[k]-b[k]) < 1e-12 for k in a)
    return same(r1.reactants, r2.products) and same(r1.products, r2.reactants)


def reaction_block(rxn, index, species_names, pressure_pa=None, rtype_suffix="", thirdBodyCase="T"):
    only_one = _is_one_third_body(rxn)
    prefix = "reversible" if rxn.reversible else "irreversible"
    rt = rxn.reaction_type

    if rt == "Arrhenius":
        body = _base_block(f"{prefix}Arrhenius{rtype_suffix}", rxn, rxn.rate)

    elif rt == "three-body-Arrhenius":
        if rxn.input_data.get("type") == "three-body":
            body = _base_block(f"{prefix}{thirdBodyCase}hirdBodyArrhenius{rtype_suffix}", rxn, rxn.rate)
            body += _third_body_block(rxn.third_body.efficiencies, species_names,
                                      must_return=not rxn.third_body.efficiencies, only_one=only_one)
        else:
            body = _base_block(f"{prefix}Arrhenius{rtype_suffix}", rxn, rxn.rate)

    elif rt in ("falloff-Troe", "falloff-Lindemann"):
        suffix_type = "Troe" if rt == "falloff-Troe" else "Lindemann"
        body = [f"        type            {prefix}Arrhenius{suffix_type}FallOff{rtype_suffix};",
                f"        reaction        \"{_fmt_eq(rxn)}\";"]
        body += _k_block("k0", rxn.rate.low_rate) + _k_block("kInf", rxn.rate.high_rate)
        if rt == "falloff-Troe":
            fc = rxn.rate.falloff_coeffs
            body += (_F_block(*fc, 4.5036e15) if len(fc) == 3 else _F_block(*fc))
        else:
            body += _F_block()
        body += _third_body_block(rxn.third_body.efficiencies, species_names,
                                  pure=False, must_return=not rxn.third_body.efficiencies, only_one=only_one)

    elif rt == "pressure-dependent-Arrhenius":
        if pressure_pa is None:
            body = _base_block(f"{prefix}ArrheniusPLOG{rtype_suffix}", rxn, rxn.rate.rates[0][1])
            body += ["        ArrheniusData", "        ("]
            body += [f"            ({fmt(p)}  {fmt(A)} {fmt(b)} {fmt(Ta)})"
                     for p, r in rxn.rate.rates for A, b, Ta in [_ap(r)]]
            body += ["        )", "        ;"]
        else:
            try:
                A, beta, Ta = _arrhenius_at_pressure(rxn.rate.rates, pressure_pa)
                rate = ct.Arrhenius(A, beta, Ta * ct.gas_constant)
            except ValueError as e:
                print(f"Warning: {e}. Using nearest pressure.")
                rate = min(rxn.rate.rates, key=lambda pr: abs(pr[0] - pressure_pa))[1]
            body = _base_block(f"{prefix}Arrhenius{rtype_suffix}", rxn, rate)

    else:
        raise NotImplementedError(f"Reaction type {rt} not supported")

    return "\n".join([f"    un-named-reaction-{index}", "    {"] + body + ["    }"])


def combined_reaction_block(forward, reverse, index, species_names, rtype_suffix="", thirdBodyCase="T"):
    rt = forward.reaction_type
    if rt == "Arrhenius":
        rtype = f"nonEquilibriumReversibleArrhenius{rtype_suffix}"
    elif rt == "three-body-Arrhenius":
        rtype = f"nonEquilibriumReversible{thirdBodyCase}hirdBodyArrhenius{rtype_suffix}"
    else:
        raise NotImplementedError(f"Non-equilibrium reversible for {rt} not supported")

    lines = [f"    un-named-reaction-{index}", "    {",
             f"        type            {rtype};",
             f"        reaction        \"{_fmt_eq(forward)}\";"]

    for label, rxn in (("forward", forward), ("reverse", reverse)):
        A, beta, Ta = _ap(rxn.rate)
        lines += [f"        {label}", "        {",
                  f"            A               {fmt(A)};",
                  f"            beta            {fmt(beta)};",
                  f"            Ta              {fmt(Ta)};"]
        if rt == "three-body-Arrhenius":
            lines += _third_body_block(rxn.third_body.efficiencies, species_names,
                                       must_return=not rxn.third_body.efficiencies)
        lines.append("        }")

    lines.append("    }")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def convert_from_chemkin_to_openfoam(ck_input_dir, ck_mech_file, ck_thermo_file,
                                      ck_transport_file, output_dir,
                                      permissive=False, pressure_atm=None):
    os.makedirs(output_dir, exist_ok=True)

    ck_mech      = os.path.join(ck_input_dir, ck_mech_file)
    ck_thermo    = os.path.join(ck_input_dir, ck_thermo_file)
    ck_transport = os.path.join(ck_input_dir, ck_transport_file)
    yaml_path    = os.path.join(output_dir, "chem.yaml")

    for path in (ck_mech, ck_thermo, ck_transport):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    chemkin_to_yaml(ck_mech, ck_thermo, ck_transport, yaml_path, permissive)
    gas = ct.Solution(yaml_path)

    print("\nGenerating Sutherland As/Ts from Cantera transport data...\n")
    default_tp, table_tp = generate_transport_from_cantera(gas, output_dir)

    header_lines, species_names = writeSpecies(gas)
    writeThermo(gas, default_tp, table_tp, output_dir, species_names, header_lines)

    elem_names = gas.element_names
    with open(os.path.join(output_dir, "elements"), "w") as f:
        f.write("\n".join(["elements", str(len(elem_names)), "("] + elem_names + [")", ";"]) + "\n")

    writeReactions(gas, species_names, output_dir, pressure_atm)

    print(f"\nConversion finished. Output: {output_dir}")


if __name__ == "__main__":
    args = ask_user_settings()
    convert_from_chemkin_to_openfoam(*args)