import os
import re
import math
import shutil
import itertools
import subprocess

import cantera as ct
import numpy as np
from scipy.optimize import curve_fit

R_CAL = 1.987
fmt = lambda v: f"{v:.9g}"

# Diffusion-coefficient fit range/resolution — see README (not user-configurable)
DIFFUSION_FIT_T_MIN = 300.0
DIFFUSION_FIT_T_MAX = 3000.0
DIFFUSION_FIT_N_POINTS = 60


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

    permissive = False
    p_str = input("Pressure for PLOG reactions in atm, empty = keep PLOG format: ").strip()
    pressure_atm = float(p_str) if p_str else None

    of_ver_str = input("Target OpenFOAM major version [8-13] (default 13): ").strip()
    of_version = int(of_ver_str) if of_ver_str else 13

    # ── Diffusion model selection ───────────────────────────────────────────────
    print(
        "\nSpecies diffusion model for thermophysicalTransport:\n"
        "  1) None — unity Lewis number (no differential diffusion, skip this step)\n"
        "  2) Simple — FickianFourier, one mixture-averaged D per species\n"
        "  3) Full  — MaxwellStefanFourier, full binary D_ij matrix\n"
    )
    choice = input("Select [1/2/3] (default 1): ").strip() or "1"
    diffusion_model = {"1": "none", "2": "fickian", "3": "maxwell_stefan"}.get(choice, "none")

    if diffusion_model in ("fickian", "maxwell_stefan") and of_version < 9:
        print(
            f"\nWarning: FickianFourier/MaxwellStefanFourier require OpenFOAM >= 9 "
            f"(target is {of_version}). Disabling diffusion generation."
        )
        diffusion_model = "none"

    diffusion_settings = {}
    if diffusion_model == "maxwell_stefan":
        # MaxwellStefanFourier is laminar-only in OpenFOAM — no RAS/LES variant exists,
        # so there's nothing to ask.
        diffusion_settings = dict(
            sim_type="laminar",
            T_min=DIFFUSION_FIT_T_MIN,
            T_max=DIFFUSION_FIT_T_MAX,
            n_points=DIFFUSION_FIT_N_POINTS,
        )

    elif diffusion_model == "fickian":
        sim_str = input(
            "  Simulation type for thermophysicalTransport [laminar/RAS/LES] "
            "(default laminar): "
        ).strip().lower() or "laminar"
        sim_type = {"laminar": "laminar", "ras": "RAS", "les": "LES"}.get(sim_str, "laminar")

        print(
            "  Reference composition for the mixture-averaged coefficients:\n"
            "    1) Equimolar mixture of all species (default)\n"
            "    2) Custom mole fractions, e.g. 'CH4:1,O2:2,N2:7.52'"
        )
        ref_choice = input("  Select [1/2] (default 1): ").strip() or "1"
        ref_composition = None
        if ref_choice == "2":
            ref_composition = input("  Enter mole fractions: ").strip() or None

        diffusion_settings = dict(
            sim_type=sim_type,
            ref_composition=ref_composition,
            T_min=DIFFUSION_FIT_T_MIN,
            T_max=DIFFUSION_FIT_T_MAX,
            n_points=DIFFUSION_FIT_N_POINTS,
        )
        if sim_type != "laminar":
            prt_str = input("  Turbulent Prandtl number Prt (default 0.85): ").strip()
            sct_str = input("  Turbulent Schmidt number Sct (default 0.7): ").strip()
            diffusion_settings["Prt"] = float(prt_str) if prt_str else 0.85
            diffusion_settings["Sct"] = float(sct_str) if sct_str else 0.7

    print(f"\nInput summary:")
    for k, v in zip(
        ["Chemkin dir", "Mech", "Thermo", "Transport", "Output", "Permissive", "PLOG pressure",
         "OF version", "Diffusion model"],
        [ck_input_dir, ck_mech_file, ck_thermo_file, ck_transport_file,
         output_dir, permissive, pressure_atm, of_version, diffusion_model]
    ):
        print(f"  {k:<20}: {v}")

    if input("\nStart conversion? [Y/n]: ").strip().lower() in ("n", "no"):
        print("Conversion cancelled.")
        exit(0)

    return (ck_input_dir, ck_mech_file, ck_thermo_file, ck_transport_file, output_dir,
            permissive, pressure_atm, of_version, diffusion_model, diffusion_settings)


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


# ── FoamFile header (parameterised by `object` name) ──────────────────────────

def _foam_header(object_name):
    return f'''\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  13
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
'''


_FOAM_HEADER = _foam_header("transportProperties")


# ── Sutherland transport fitting ──────────────────────────────────────────────

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


# ── Diffusion coefficient fitting (MaxwellStefanFourier / FickianFourier) ─────

POLY_DEGREE = 4  # binaryDiffusionCoefficient == Polynomial<5>, do not change


def _fit_D_polynomial(D_vals, T_array, p_ref):
    # D(T) = T^1.5*(c0+c1*lnT+...+c4*lnT^4)/p
    lnT = np.log(T_array)
    y = D_vals * p_ref / (T_array ** 1.5)
    coeffs_high_to_low = np.polyfit(lnT, y, POLY_DEGREE)
    coeffs = coeffs_high_to_low[::-1]
    y_fit = np.polyval(coeffs_high_to_low, lnT)
    D_fit = y_fit * (T_array ** 1.5) / p_ref
    rel_err = np.abs(D_fit - D_vals) / np.maximum(D_vals, 1e-30)
    return coeffs, rel_err.max()


def _sample_D_over_T(T_min, T_max, n_points, getter):
    # getter(T) -> D array at that temperature; stacks into one (n_points, ...) array
    T_array = np.linspace(T_min, T_max, n_points)
    return T_array, np.array([getter(T) for T in T_array])


def fit_binary_diffusion_polynomials(gas, T_min=300.0, T_max=3000.0, n_points=60,
                                      p_ref=None):
    # {(sp_i, sp_j): [c0..c4]} for all i<=j, incl. self-diffusion (composition-independent)
    p_ref = p_ref or ct.one_atm
    T0, P0, X0 = gas.T, gas.P, gas.X.copy()
    gas.transport_model = "mixture-averaged"
    species_names = gas.species_names

    def getter(T):
        gas.TP = T, p_ref
        return gas.binary_diff_coeffs  # [m^2/s], composition-independent

    T_array, D_matrices = _sample_D_over_T(T_min, T_max, n_points, getter)

    pair_coeffs, max_rel_err_dict = {}, {}
    for i, j in itertools.combinations_with_replacement(range(len(species_names)), 2):
        coeffs, max_err = _fit_D_polynomial(D_matrices[:, i, j], T_array, p_ref)
        key = (species_names[i], species_names[j])
        pair_coeffs[key], max_rel_err_dict[key] = coeffs, max_err

    gas.TPX = T0, P0, X0  # restore state for downstream thermo/reaction writers
    return pair_coeffs, max_rel_err_dict


def fit_mixture_diffusion_polynomials(gas, T_min=300.0, T_max=3000.0, n_points=60,
                                       p_ref=None, ref_composition=None):
    # {sp: [c0..c4]}, one mixture-averaged D per species (Wilke rule, depends on ref_composition)
    p_ref = p_ref or ct.one_atm
    T0, P0, X0 = gas.T, gas.P, gas.X.copy()
    gas.transport_model = "mixture-averaged"
    species_names = gas.species_names
    gas.X = ref_composition or {sp: 1.0 for sp in species_names}  # equimolar default

    def getter(T):
        gas.TP = T, p_ref
        return gas.mix_diff_coeffs  # [m^2/s], depends on the composition set above

    T_array, D_matrix = _sample_D_over_T(T_min, T_max, n_points, getter)

    species_coeffs, max_rel_err_dict = {}, {}
    for i, sp in enumerate(species_names):
        coeffs, max_err = _fit_D_polynomial(D_matrix[:, i], T_array, p_ref)
        species_coeffs[sp], max_rel_err_dict[sp] = coeffs, max_err

    gas.TPX = T0, P0, X0
    return species_coeffs, max_rel_err_dict


def _print_fit_errors(err_dict, label):
    worst = sorted(err_dict.items(), key=lambda kv: -kv[1])[:5]
    print(f"\nWorst-fit {label}:")
    for key, err in worst:
        key_str = "-".join(key) if isinstance(key, tuple) else key
        print(f"  {key_str}: {err*100:.4f}%")
    errs = np.array(list(err_dict.values()))
    print(f"Max relative error: {errs.max()*100:.4f}%")
    print(f"Mean relative error: {errs.mean()*100:.4f}%")


def _diffusion_dict_text(model, T_min, T_max, p_ref, n_species, entry_coeffs,
                          mixture_averaged=False, implicit_heat_flux=None,
                          sim_type="laminar", Prt=None, Sct=None):
    # entry_coeffs: {key: [c0..c4]}, key is "spA-spB" (D) or "sp" (Dm)
    fickian_like = model in ("FickianFourier", "FickianEddyDiffusivity")
    lines = [_foam_header("thermophysicalTransport")]
    lines.append(f"// {n_species} species, {len(entry_coeffs)} entries, fit {T_min:.0f}-{T_max:.0f} K")
    lines.append("")
    lines.append(f"simulationType {sim_type};")
    lines.append("")
    lines.append(sim_type)
    lines.append("{")
    lines.append(f"    model  {model};")
    if fickian_like:
        lines.append(f"    mixtureDiffusionCoefficients {'yes' if mixture_averaged else 'no'};")
    if sim_type != "laminar":
        lines.append(f"    Prt {fmt(Prt)};")
        lines.append(f"    Sct {fmt(Sct)};")
    if implicit_heat_flux is not None:
        val = "true" if implicit_heat_flux else "false"
        lines.append(f"    implicitHeatFlux {val};")
    lines.append("")
    lines.append("    Dm" if (fickian_like and mixture_averaged) else "    D")
    lines.append("    {")
    for key, c in entry_coeffs.items():
        coeffs_str = " ".join(f"{v:.8e}" for v in c)
        lines.append(f"        {key}")
        lines.append("        {")
        lines.append("            type   binaryDiffusionCoefficient;")
        lines.append(f"            coeffs ({coeffs_str});")
        lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("// ************************************************************************* //")
    return "\n".join(lines)


def generate_maxwellstefan_transport(gas, output_dir, T_min=300.0, T_max=3000.0,
                                      n_points=60, implicit_heat_flux=None,
                                      sim_type="laminar", Prt=0.85, Sct=0.7):
    p_ref = ct.one_atm
    pair_coeffs, max_rel_err_dict = fit_binary_diffusion_polynomials(
        gas, T_min=T_min, T_max=T_max, n_points=n_points, p_ref=p_ref
    )
    _print_fit_errors(max_rel_err_dict, "binary diffusion pairs")

    # MaxwellStefanFourier is laminar-only; no RAS/LES counterpart exists, so
    # turbulent cases fall back to FickianEddyDiffusivity reusing the same
    # binary D_ij coefficients (mixtureDiffusionCoefficients no).
    model = "MaxwellStefanFourier" if sim_type == "laminar" else "FickianEddyDiffusivity"

    entry_coeffs = {f"{a}-{b}": c for (a, b), c in pair_coeffs.items()}
    text = _diffusion_dict_text(
        model, T_min, T_max, p_ref, gas.n_species, entry_coeffs,
        mixture_averaged=False, implicit_heat_flux=implicit_heat_flux,
        sim_type=sim_type, Prt=Prt, Sct=Sct,
    )
    out_path = os.path.join(output_dir, "thermophysicalTransport")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Wrote: {out_path}")


def generate_fickian_transport(gas, output_dir, T_min=300.0, T_max=3000.0,
                                n_points=60, ref_composition=None,
                                implicit_heat_flux=None,
                                sim_type="laminar", Prt=0.85, Sct=0.7):
    species_coeffs, max_rel_err_dict = fit_mixture_diffusion_polynomials(
        gas, T_min=T_min, T_max=T_max, n_points=n_points,
        ref_composition=ref_composition,
    )
    _print_fit_errors(max_rel_err_dict, "mixture-averaged diffusion species")
    if not ref_composition:
        print("Note: used an equimolar reference composition.")

    model = "FickianFourier" if sim_type == "laminar" else "FickianEddyDiffusivity"
    text = _diffusion_dict_text(
        model, T_min, T_max, ct.one_atm, gas.n_species, species_coeffs,
        mixture_averaged=True, implicit_heat_flux=implicit_heat_flux,
        sim_type=sim_type, Prt=Prt, Sct=Sct,
    )
    out_path = os.path.join(output_dir, "thermophysicalTransport")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Wrote: {out_path}")


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
                                      permissive=False, pressure_atm=None,
                                      of_version=13,
                                      diffusion_model="none",
                                      diffusion_settings=None):
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

    # ── Diffusion transport ──────────────────────────────────────────────────────
    settings = diffusion_settings or {}
    if diffusion_model in ("maxwell_stefan", "fickian") and of_version < 9:
        print(f"\nWarning: {diffusion_model} needs OpenFOAM >= 9 (target {of_version}). Skipping.")
        diffusion_model = "none"

    if diffusion_model == "maxwell_stefan":
        print("\nGenerating MaxwellStefanFourier binary diffusion coefficients...\n")
        generate_maxwellstefan_transport(gas, output_dir, **settings)
    elif diffusion_model == "fickian":
        print("\nGenerating FickianFourier mixture-averaged diffusion coefficients...\n")
        generate_fickian_transport(gas, output_dir, **settings)
    else:
        print("\nDiffusion model: none (unity Lewis) — skipping thermophysicalTransport.")

    print(f"\nConversion finished. Output: {output_dir}")


if __name__ == "__main__":
    args = ask_user_settings()
    convert_from_chemkin_to_openfoam(*args)