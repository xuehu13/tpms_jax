# r43: diverse_28 initial occupation experiment (frozen before binary solve)

L10mm, t0.5mm, E10MPa, nu0.3, eta1e-4, N32 HEX27, macro Hzz=-1e-4, XYZ periodic fluctuations and zero macro transverse strain. Reuse r41 smooth state and shell K=3.7748837447547623N/mm. No new Abaqus job, production change, path or design AD.

Own selected region: cells with any original Gauss phi in [0.01,0.99], ranked by saved smooth cell energy; smallest prefix covering 80% candidate energy. Selection is 3896/32768 cells, representing 73.7772% smooth full energy. It is NOT diverse_04's old 1941-cell region. Outside selection is not densely checked.

One original27 binary solve, phi=1[d<=0.25mm]; eta remains. Before any matched integration pair, fixed smooth/binary states are sampled at 8 and 12 points/axis under both smooth/binary fields. Each selected energy, occupied volume and occupied squared-distance moment must change <=1% (denominator dense12). Constant-floor energies must reproduce original27 <=1e-10; saved total energy reproduction <=1e-9. Failed gates stop the pair; no added levels or tolerance changes.

Passed gate permits only ONE smooth/binary matched selected12/rest27 pair, retaining r34 operator symmetry/translation/reconstruction, residual<=1e-8, work consistency<=1e-6 and periodic checks. Budget 900s per major attempt and 2700s active total, max6000 PCG iterations; no retries. Effect relative to same-rule smooth K/U; shell difference relative to fixed r41 shell. Limits are partial integration coverage, linearized zero-state scope, and no hard-threshold geometry AD certification.

Input repair: initial fixed-check stopped before dense sampling because the r6 cache has no surface arrays. Preserve original log/source/config in before_geometry_input_repair/. Use original r5 archived surface arrays with <=1e-12 original distance reproduction; physical parameters, selection, integration levels and gates unchanged. No repeated equilibrium solve.
