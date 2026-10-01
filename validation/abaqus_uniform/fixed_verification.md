# First actual Abaqus 2026 uniform baseline verification

The user ran `uniform_xy_fixed.inp` in Abaqus/Standard 2026 on 2026-10-01.
The agent read the completed ODB without submitting or rerunning any analysis.
The status file reports `THE ANALYSIS HAS COMPLETED SUCCESSFULLY`; the message
file reports zero error messages.

The first actual ODB read exposed NumPy scalar values returned by the Abaqus
API. JSON cannot serialize `numpy.bool_`. The extractor now converts stress
components and displacement coordinates to Python floats and check results to
Python bool before serialization. The repaired extractor was tested against
this actual ODB and produced `status=ok` with all 12 checks passing.

Measured sigma_zz and QZ.RF3: -0.13461539149284363.
Measured ALLSE: 0.0006730768945999444.
Measured integrated volume: 1.0, with 512 integration points.
Maximum physical affine-displacement error: 2.2351741811588166e-10.
These agree with the analytic uniform fixed-lateral response within the
predeclared 1e-6 relative acceptance target. No claim is made here about either
relaxed case, nonuniform material fields, or binary Gyroid geometry.

The original preparation manifest and README describe the state before any
Abaqus solve. This later record and `measured_fixed.json` document the actual
fixed-case verification. The ODB itself remains in the user's job directory
and is not committed.
