# Ball Sports Simulation - Modular Architecture

This project has been refactored into a modular architecture to improve maintainability, extensibility, and reusability. The modular design makes it easier to extend the simulation to other ball sports beyond table tennis.

## Architecture Overview

The codebase is organized into the following modules:

### Core Modules

1. **`constants.py`** - All physical constants and parameters
   - Physical properties (gravity, air density, ball characteristics)
   - Table and racket specifications
   - Simulation parameters and default values

2. **`ball_types.py`** - Data structures and type definitions
   - BallState, RacketState, Table, Net classes
   - Enums for rubber types, events, players
   - Configuration dataclasses

3. **`physics.py`** - Physics calculations and collision handling
   - Aerodynamic forces (drag, Magnus effect)
   - Numerical integration (RK4 method)
   - Collision detection and response
   - Racket movement physics

4. **`racket_control.py`** - Racket control and stroke logic
   - Stroke parameter definitions
   - Player decision-making for hitting
   - Racket positioning and movement

5. **`scenarios.py`** - Predefined simulation scenarios
   - Table and net setup
   - Scenario configurations (serve, smash, custom)
   - Initial condition definitions

### Engine Modules

6. **`simulation.py`** - Core simulation engine
   - Main simulation loop
   - Event tracking and logging
   - Result data collection

7. **`visualization.py`** - Output and visualization
   - Data export to CSV
   - 3D plotting and animation
   - Summary reporting

8. **`main.py`** - Command-line interface
   - Argument parsing
   - Output directory management
   - Main execution flow

## Key Features of the Modular Design

### Extensibility
- **Ball Types**: Easy to add new ball types by extending `constants.py` and `ball_types.py`
- **Sports**: Different sports can be implemented by creating new scenario modules
- **Physics**: Physics models can be extended or modified in `physics.py`

### Maintainability
- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Clear Interfaces**: Well-defined APIs between modules
- **Type Safety**: Comprehensive type hints throughout

### Reusability
- **Shared Components**: Physics and visualization modules can be reused across sports
- **Configurable**: Parameters are centralized and easily modifiable
- **Import-Friendly**: Clean import structure for external use

## Usage

### Running Simulations

```bash
# Basic serve simulation
python pingpong_main.py --scenario serve

# Custom scenario with specific parameters
python pingpong_main.py --scenario custom --pos -1.2 0 0.9 --vel 8 0 3 --omega 0 150 0

# Output to custom directory
python pingpong_main.py --scenario smash --output-dir ./my_results
```

### Extending to New Ball Sports

To add a new ball sport (e.g., tennis, soccer, basketball):

1. **Update Constants** (`constants.py`):
   - Add ball-specific physical parameters
   - Define playing field dimensions

2. **Extend Types** (`ball_types.py`):
   - Add sport-specific data structures if needed
   - Define relevant enums (e.g., court positions, ball types)

3. **Create Physics Models** (`physics.py`):
   - Implement sport-specific aerodynamic models
   - Add collision detection for new obstacles

4. **Define Scenarios** (`scenarios.py` or new module):
   - Create sport-specific initial conditions
   - Define typical play scenarios

5. **Update Main Interface** (`main.py`):
   - Add command-line options for the new sport
   - Integrate scenario loading

## Example: Adding Tennis Support

```python
# In constants.py - add tennis-specific constants
BALL_RADIUS_TENNIS = 0.033  # 6.7cm diameter
COURT_LENGTH_TENNIS = 23.77  # meters
# ... etc

# In scenarios.py - add tennis scenarios
def create_tennis_serve_scenario() -> Tuple[BallState, ...]:
    # Tennis-specific serve setup
    pass

# In pingpong_main.py - add tennis option
parser.add_argument('--sport', choices=['pingpong', 'tennis'], default='pingpong')
```

## Dependencies

- numpy: Numerical computations
- matplotlib: Visualization and plotting
- pandas: Data manipulation and CSV export

## Benefits of Modularization

1. **Easier Testing**: Each module can be tested independently
2. **Parallel Development**: Multiple developers can work on different modules
3. **Code Reuse**: Common functionality is centralized
4. **Future-Proof**: Easy to extend to new sports and features
5. **Maintainability**: Clear structure makes bugs easier to locate and fix

## File Structure

```
├── constants.py          # Physical constants and parameters
├── ball_types.py         # Data structures and enums
├── physics.py           # Physics calculations and collisions
├── racket_control.py    # Racket control and stroke logic
├── scenarios.py         # Predefined scenarios
├── simulation.py        # Core simulation engine
├── visualization.py     # Output and plotting
├── main.py             # Command-line interface
└── MODULAR_README.md   # This documentation
```

This modular architecture provides a solid foundation for expanding the simulation to multiple ball sports while maintaining clean, maintainable code.


