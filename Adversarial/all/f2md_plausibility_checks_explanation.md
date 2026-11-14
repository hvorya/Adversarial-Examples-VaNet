# Line-by-line and function-by-function guide to `f2md_plausibility_checks.py`

This guide follows the Python file's line numbers. Lines are grouped when consecutive lines perform one small operation; every executable line is covered in the group descriptions.

## 1. Module header and imports (lines 1–20)

- **Lines 1–14 — module docstring:** Describes the purpose and provenance of this file. It says this is a Python adaptation of the F2MD C++ files `LegacyChecks.cc`, `ExperiChecks.cc`, and `CaTChChecks.cc`. It warns that the C++ implementation depends on Veins/OMNeT++, `mdmLib`, `LinkControl`, and Kalman objects. The module aims to provide standalone versions of the geometry and simpler plausibility tests. It also documents the score convention: usually `1` is plausible and `0` is implausible, while some checks give continuous scores.
- **Line 16:** Imports `dataclass`, a decorator that automatically creates common class methods (such as an initializer) for a data class.
- **Line 17:** Imports mathematical operations:
  - `atan2`: angle of a vector.
  - `cos`, `sin`: trigonometry.
  - `hypot`: Euclidean length.
  - `fabs`: absolute value (not used in the current file).
  - `radians`, `degrees`: angle unit conversion.
  - `pi`: the constant π.
  - `exp`: exponential (not used in the current file).
- **Line 18:** Imports typing helpers to document expected data types. `Iterable` and `Sequence` are unused here.
- **Line 20:** Defines `Point` as a type alias for a pair of floating-point values `(x, y)`.

## 2. Coordinate and geometry helper functions

### `xy(p)` — lines 23–26
Converts a point into a standard two-number tuple.
- **Line 23:** Declares the function and its return type.
- **Line 24:** Checks whether `p` has an `.x` attribute, as Veins coordinate objects do.
- **Line 25:** If so, returns its `.x` and `.y` values converted to floats.
- **Line 26:** Otherwise, assumes `p` is indexable (like a tuple/list) and converts `p[0]`, `p[1]` to floats.

### `distance(a, b)` — lines 29–31
Computes straight-line distance between two points.
- **Line 29:** Declares the function.
- **Line 30:** Converts both points to `(x, y)` and assigns their coordinates.
- **Line 31:** Uses `hypot(ax-bx, ay-by)`, equivalent to `sqrt((ax-bx)^2 + (ay-by)^2)`. The result is in the coordinate system's distance unit (usually metres).

### `heading_angle(vector)` — lines 34–36
Computes a vector's orientation.
- **Line 34:** Declares the function.
- **Line 35:** Extracts its x and y components.
- **Line 36:** `atan2(y, x)` determines the angle, then `degrees(...)` converts radians to degrees. Its result lies in the usual `[-180°, 180°]` interval.

### `angle_difference(a, b)` — lines 39–41
Returns the shortest absolute angular difference, accounting for wrap-around.
- **Line 40:** Docstring states the behavior.
- **Line 41:** Applies modulo arithmetic to normalize the difference to `[-180°, 180°]`, then takes its absolute value. Thus, e.g. 359° and 1° differ by 2°, not 358°.

### `circle_overlap_area(r1, r2, d)` — lines 44–55
Calculates the overlap area of two circles, whose radii are `r1` and `r2` and whose centers are `d` apart.
- **Line 45:** Describes the geometric purpose.
- **Line 46:** Clamps radii and distance to nonnegative values.
- **Lines 47–48:** If circles do not touch (`d >= r1+r2`), their overlap area is zero.
- **Lines 49–50:** If one circle lies fully inside the other (`d <= abs(r1-r2)`), overlap is the area of the smaller circle.
- **Lines 51–52:** Computes the two central angles using the circle-intersection formula. `acos_clamped` guards against tiny floating-point errors.
- **Line 53:** Adds the two sector areas.
- **Line 54:** Subtracts the triangular portions to produce the lens-shaped overlap area.
- **Line 55:** Returns the computed area.

### `acos_clamped(x)` — lines 58–60
- **Line 58:** Declares helper.
- **Line 59:** Imports `acos` locally.
- **Line 60:** Clamps `x` into `[-1, 1]` before taking inverse cosine. This avoids a domain error caused by floating-point rounding.

### `sqrt_nonnegative(x)` — lines 63–65
- **Line 63:** Declares helper.
- **Line 64:** Imports square root locally.
- **Line 65:** Clamps `x` to zero or above before calculating its square root, preventing errors from a slightly negative numerical value.

### `circle_circle_factor(d, r1, r2, allowable)` — lines 68–79
A simplified confidence-aware plausibility score. `d` is measured distance; `r1` and `r2` represent uncertainty radii; `allowable` is the maximum acceptable nominal distance.
- **Lines 69–73:** Explain that this is an approximation of the project's `mdmLib.CircleCircleFactor`, not an exact port.
- **Line 74:** Creates an outer threshold by adding the allowed distance and both uncertainty radii, clamping each to nonnegative.
- **Lines 75–76:** Returns full plausibility (`1`) if the distance is within the allowable distance adjusted by the difference between uncertainty radii.
- **Lines 77–78:** Returns `0` if distance reaches/exceeds the outer threshold.
- **Line 79:** For distances between those bounds, linearly interpolates a score between 1 and 0. The small `1e-12` denominator floor prevents division by zero.

## 3. Parameter configuration (lines 82–99)

### `CheckParameters` data class
- **Line 82:** `@dataclass` asks Python to generate standard data-class methods.
- **Line 83:** Declares the parameter container.
- **Lines 84–98:** Define default thresholds used by the checks:
  - **84 `MAX_PROXIMITY_RANGE_L`:** longitudinal/large proximity limit.
  - **85 `MAX_PROXIMITY_RANGE_W`:** width-related proximity limit.
  - **86 `MAX_PROXIMITY_DISTANCE`:** distance threshold used for nearby-object plausibility.
  - **87 `MAX_PLAUSIBLE_RANGE`:** maximum sender/receiver range.
  - **88 `MAX_NON_ROUTE_SPEED`:** speed below which off-road position is tolerated.
  - **89 `MAX_DISTANCE_FROM_ROUTE`:** allowed distance from a road/route.
  - **90 `MAX_BEACON_FREQUENCY`:** minimum time interval used by the beacon timing test (despite the name, it is represented as a time threshold).
  - **91 `POS_HEADING_TIME`:** time window in which position and heading are compared.
  - **92 `MAX_DELTA_INTER`:** time window used to discount intersection suspicion.
  - **93 `MAX_SA_RANGE`:** range threshold for sudden-appearance detection.
  - **94 `MAX_TIME_DELTA`:** time gate for the max position-speed consistency check.
  - **95 `MAX_HEADING_CHANGE`:** allowed angular difference.
  - **96 `MAX_KALMAN_TIME`:** Kalman filter time threshold.
  - **97 `MAX_KALMAN_POS_RANGE`:** Kalman position threshold.
  - **98 `MAX_KALMAN_SPEED_RANGE`:** Kalman speed threshold.
- **Line 99:** Blank separator.

These are illustrative defaults in the Python port. For meaningful experiments, use the parameter values from the F2MD configuration files.

## 4. Main class and initializer (lines 101–114)

### `F2MDChecks`
- **Line 101:** Defines the class that groups the check methods.
- **Line 102:** Class docstring.

### `__init__(...)`
- **Line 104:** Starts the constructor. `params` is optional; `max_speed`, `max_accel`, and `max_decel` set motion limits; `route_distance` is a callback for road distance; `dynamic_map` stores other vehicle positions; `my_id` identifies the receiving vehicle.
- **Lines 105–107:** Continue the arguments and their defaults/types.
- **Line 108:** Stores supplied parameters or creates default `CheckParameters()`.
- **Line 109:** Stores maximum plausible speed.
- **Line 110:** Stores maximum plausible acceleration.
- **Line 111:** Stores maximum plausible deceleration.
- **Line 112:** Stores the callback used to compute distance from a position to the nearest route.
- **Line 113:** Stores the dynamic map, or an empty dictionary if none was supplied.
- **Line 114:** Stores this vehicle's identifier so it can be excluded from neighbor comparisons.

## 5. Plausibility check methods

### `ProximityPlausibilityCheck(...)` — lines 116–131
Checks whether a reported vehicle position is plausible in the receiver's nearby spatial region and in relation to the dynamic map.
- **Line 116:** Function declaration; `myHeading` is accepted but not used by this Python version.
- **Line 117:** Short alias `p` for the parameters object.
- **Line 118:** Calculates distance from the test vehicle to the receiver.
- **Line 119:** Extracts the two positions' coordinates.
- **Line 120:** Calculates the absolute angle of the relative-position vector.
- **Line 121:** Proceeds only if distance is less than the configured long-range proximity threshold.
- **Lines 122–124:** Determines whether the test position is within a width/lane-like region. It uses either a simple near-distance condition or an angle-dependent width boundary. `max(1e-12, ...)` avoids division by zero.
- **Line 125:** Enters the nearby-object check if the point falls in that region.
- **Lines 126–127:** Finds the smallest distance from the test position to any entry in `dynamic_map`, excluding the receiver itself. If there are no eligible entries, uses `9999`.
- **Lines 128–129:** Comments distinguish the graded CaTCh/Experi score from the binary Legacy version.
- **Line 130:** Converts nearest-neighbor distance to a graded score: nearer neighbor means a larger score under this implementation's formula.
- **Line 131:** Returns 1 if the earlier region/distance conditions were not met.

### `RangePlausibilityCheck(...)` — lines 133–141
Checks whether two vehicles are within the maximum plausible communication/position range.
- **Lines 133–135:** Function signature includes both positions, optional uncertainty values, and a flag to choose confidence-aware scoring.
- **Line 136:** Calculates nominal distance.
- **Lines 137–138:** In binary mode, returns 1 if distance is strictly less than `MAX_PLAUSIBLE_RANGE`, otherwise 0.
- **Lines 139–141:** In confidence-aware mode, calls the approximation `circle_circle_factor` using x components of the confidence radii and the configured range.

### `PositionPlausibilityCheck(...)` — lines 143–171
Checks whether a moving vehicle's reported location is reasonably close to a road/route. Low-speed vehicles are exempt.
- **Lines 143–145:** Declares position, speed, position-confidence radius, and speed-confidence arguments.
- **Line 146:** Alias for the parameters.
- **Line 147:** Estimates a conservative speed by subtracting speed uncertainty, then floors it at zero.
- **Lines 148–149:** If that speed is at/below `MAX_NON_ROUTE_SPEED`, returns plausible (1), since a stopped/slow vehicle may be off-road.
- **Lines 150–151:** Requires a `route_distance` callback; raises an explanatory error if missing.
- **Line 152:** Documents callback contract: it returns distance to the nearest road.
- **Line 153:** Gets the position uncertainty radius from the x component.
- **Lines 154–155:** If no uncertainty radius exists, checks just the reported point against `MAX_DISTANCE_FROM_ROUTE`.
- **Line 156:** Chooses a radial sampling resolution, with a minimum of 1 unit.
- **Line 157:** Sets the increment between sampled radii to 1.1 times the resolution.
- **Line 158:** Initializes number of failed samples and total sample count; starts total at 1 for the central point.
- **Lines 159–160:** Tests the reported center point; if too far from the road, increments failed count.
- **Line 161:** Starts sampling outward from the first radius.
- **Line 162:** Repeats sampling while radius is smaller than the confidence radius.
- **Line 163:** Chooses number of angular samples around that ring, at least one.
- **Line 164:** Gets center coordinates.
- **Lines 165–166:** For each angle index, creates a sample point on the ring using cosine and sine.
- **Lines 167–168:** Counts samples that exceed the route-distance limit.
- **Line 169:** Increments the total number of samples.
- **Line 170:** Advances to the next ring radius.
- **Line 171:** Returns one minus the fraction of samples deemed off-route. This is a proportion-like score.

### `SpeedPlausibilityCheck(...)` — lines 173–174
- **Line 173:** Declares speed and optional speed confidence.
- **Line 174:** Checks whether the magnitude of speed minus its confidence is below `MAX_PLAUSIBLE_SPEED`. Returns 1 if so, else 0. This approximates the confidence-aware source behavior.

### `PositionConsistancyCheck(...)` — lines 176–184
Checks whether movement between two reported positions is possible under a maximum speed.
- **Lines 176–178:** Defines current and prior positions, time interval, their uncertainty radii, and a confidence-aware switch.
- **Line 179:** Computes distance between current and previous position.
- **Line 180:** Computes allowed displacement as maximum speed multiplied by nonnegative elapsed time.
- **Lines 181–182:** Binary mode returns 1 if measured distance is below the bound, otherwise 0.
- **Lines 183–184:** Confidence-aware mode returns the approximate continuous circle-overlap factor.

### `SpeedConsistancyCheck(...)` — lines 186–191
Checks whether change in speed is plausible under acceleration/deceleration limits.
- **Lines 186–187:** Function declaration includes current speed, old speed, elapsed time, and optional confidence values.
- **Line 188:** Calculates signed speed change.
- **Line 189:** Selects acceleration limit if speed increased; deceleration limit otherwise; scales by nonnegative time.
- **Line 190:** Adds nonnegative uncertainty values for the current and old speeds.
- **Line 191:** Returns 1 if absolute speed change does not exceed limit plus confidence allowance, else 0.

### `PositionSpeedConsistancyCheck(...)` — lines 193–199
A simplified comparison of displacement-derived speed with the reported speeds.
- **Lines 193–194:** Declares positions, current and old speeds, and time.
- **Lines 195–196:** If elapsed time is nonpositive, returns 1 to avoid division by zero.
- **Line 197:** Calculates average of current and previous speed.
- **Line 198:** Calculates observed speed as distance between positions divided by elapsed time.
- **Line 199:** Returns 1 if observed speed does not exceed the largest of the two reported magnitudes and their average; otherwise returns 0.

**Important:** This particular function is a simplified approximation in this generated Python file, not a verified exact translation of the C++ implementation.

### `PositionSpeedMaxConsistancyCheck(...)` — lines 201–206
A second, time-gated position/speed comparison.
- **Lines 201–202:** Declares its arguments.
- **Lines 203–204:** Returns 1 without checking if time is nonpositive or reaches/exceeds `MAX_TIME_DELTA`.
- **Line 205:** Calculates displacement-derived speed.
- **Line 206:** Returns 1 if observed speed is at most the greater of the two reported speed magnitudes, otherwise 0.

This also is a simplified port, not a bit-for-bit translation of the original C++ routine.

### `BeaconFrequencyCheck(...)` — lines 208–209
- **Line 208:** Declares new and old beacon-arrival timestamps.
- **Line 209:** Calculates their interval implicitly. Returns 0 if the interval is below `MAX_BEACON_FREQUENCY`, otherwise 1. Note the code uses this threshold as a minimum interval in time units.

### `PositionHeadingConsistancyCheck(...)` — lines 211–223
Checks whether the reported heading agrees with the direction of movement.
- **Lines 211–212:** Declares current heading, current and prior position, elapsed time, speed, and optional angular limit.
- **Lines 213–214:** If time is outside the configured comparison window, returns plausible (1).
- **Line 215:** Calculates displacement.
- **Lines 216–217:** If displacement is under 1 unit or speed under 1, skips the check and returns 1, because heading is unreliable while stationary/near-stationary.
- **Lines 218–219:** Calculates x/y displacement components.
- **Line 220:** Converts the displacement vector to its movement heading.
- **Line 221:** Converts the claimed heading vector to an angle.
- **Line 222:** Uses the configured heading-change limit unless the caller supplied a different limit.
- **Line 223:** Returns 1 if the shortest angular difference is within the limit; otherwise 0.

### `SuddenAppearenceCheck(...)` — lines 225–237
Flags a vehicle that appears unexpectedly far from the receiver. The function name retains the spelling used by the source code.
- **Lines 225–228:** Declares sender/receiver positions, optional confidence radii, and a confidence-aware flag.
- **Line 229:** Calculates their nominal distance.
- **Lines 230–231:** Binary mode returns 0 if sender is inside `MAX_SA_RANGE`; otherwise returns 1 (suspicious).
- **Line 232:** Reads sender's position confidence radius.
- **Line 233:** Enlarges receiver's range by receiver position uncertainty.
- **Lines 234–235:** If sender uncertainty is zero, uses a direct distance comparison.
- **Line 236:** Calculates overlap between sender's uncertainty circle and the receiver's expanded proximity circle.
- **Line 237:** Returns one minus the fraction of sender uncertainty area that overlaps the receiver range. Little overlap produces a high suspiciousness score.

### `IntersectionCheck(...)` — lines 239–266
Checks whether two oriented rectangular vehicle footprints overlap. This is a Python geometry approximation of the C++ rectangle-factor method.
- **Lines 239–240:** Declares two vehicle positions, sizes, headings, and elapsed time.
- **Lines 241–245:** Docstring describes the Separating Axis Theorem (SAT), size convention `(width, length)`, headings in degrees, and score convention.
- **Line 246:** Defines a nested helper to compute the four rectangle corners.
- **Line 247:** Reads center and width/length.
- **Line 248:** Halves dimensions to get half-width and half-length.
- **Line 249:** Converts heading angle from degrees to radians.
- **Lines 250–251:** Computes two perpendicular unit direction vectors for the rectangle's local axes.
- **Lines 252–253:** Generates four corner coordinates using combinations of positive/negative half-dimensions.
- **Line 254:** Builds the corner lists for the two vehicles.
- **Line 255:** Initializes the list of candidate projection axes.
- **Line 256:** Iterates over each rectangle.
- **Line 257:** Considers the first two edges of each rectangle (enough to get its two unique axis directions).
- **Line 258:** Extracts edge endpoint coordinates.
- **Line 259:** Builds an axis perpendicular to the edge.
- **Line 260:** Initially assumes the rectangles intersect.
- **Line 261:** Tests each candidate axis.
- **Line 262:** Projects each rectangle's corners onto that axis.
- **Lines 263–264:** If the projection intervals are separated, the rectangles do not intersect; mark false and stop checking.
- **Line 265:** Computes a time-based weight: closer timestamps give higher weight, older comparisons decay toward zero.
- **Line 266:** Returns suspicious score 0 when the rectangles intersect and the weight exceeds 0.5; otherwise returns 1.

### Kalman-related methods — lines 268–283
These methods are placeholders, not complete implementations.
- **Lines 268–269:** Comment explains the missing dependency: project-specific Kalman filter adapters.
- **Lines 270–271 — `KalmanPositionConsistancyCheck`:** Raises `NotImplementedError`. An F2MD `Kalman_SI` compatible adapter must be connected to port the position-prediction test.
- **Lines 273–274 — `KalmanPositionAccConsistancyCheck`:** Same limitation; requires `Kalman_SI`.
- **Lines 276–277 — `KalmanSpeedConsistancyCheck`:** Requires the source's speed Kalman filter adapter.
- **Lines 279–280 — `KalmanPositionSpeedConsistancyCheck`:** Requires the `Kalman_SVI` adapter.
- **Lines 282–283 — `KalmanPositionSpeedScalarConsistancyCheck`:** Requires the `Kalman_SC` adapter.

Calling one of these methods currently raises an exception by design; the Python module does not yet implement the prediction/scoring internals.

## 6. How to instantiate and use the module

```python
from f2md_plausibility_checks import F2MDChecks, CheckParameters

params = CheckParameters(
    MAX_PLAUSIBLE_RANGE=250.0,
    MAX_NON_ROUTE_SPEED=2.0,
    MAX_DISTANCE_FROM_ROUTE=15.0,
)

# route_distance(point) must return distance from point (x, y) to the nearest road.
checks = F2MDChecks(
    params=params,
    max_speed=50.0,
    max_accel=4.0,
    max_decel=8.0,
    route_distance=lambda point: 3.0,  # illustrative callback only
    dynamic_map={"vehicle_B": (12.0, 7.0), "vehicle_C": (25.0, 9.0)},
    my_id="vehicle_A",
)

range_score = checks.RangePlausibilityCheck((0, 0), (100, 0))
position_score = checks.PositionConsistancyCheck((10, 0), (0, 0), time=1.0)
heading_score = checks.PositionHeadingConsistancyCheck(
    curHeading=(1, 0), curPosition=(10, 0), oldPosition=(0, 0),
    deltaTime=0.5, curSpeed=10.0
)

print(range_score, position_score, heading_score)
```

## 7. Limitations and fidelity note

The file was created as a usable explanatory Python scaffold, not a complete, exact, tested port of all C++ source behavior. In particular:
1. `circle_circle_factor` approximates the source's `mdmLib.CircleCircleFactor`.
2. `PositionSpeedConsistancyCheck` and `PositionSpeedMaxConsistancyCheck` are simplified versions.
3. `IntersectionCheck` uses SAT rather than the source library's exact rectangle-factor implementation.
4. Kalman methods are unimplemented placeholders.
5. Some unused imports remain in the generated file.
6. The original source's score interpretation should be checked method-by-method; not all methods uniformly use `1 = plausible`, `0 = suspicious`.

For a research paper or implementation evaluation, these functions should be validated against the C++ source with test inputs and the same configured parameters.
