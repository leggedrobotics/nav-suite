# Navigation Suite Tasks

The `nav_tasks` extension provides a framework for robotic navigation tasks, including MDP components, training capabilities, and benchmarking tools.

## MDP Components

### Goal Commands
- **GoalCommand:** Samples goal positions using terrain analysis
- **ConsecutiveGoalCommand:** Generates sequences of terrain-aware goals
- **FixGoalCommand:** Provides fixed goal positions

### Reward Terms
- **near_goal_stability:** Rewards stability and low velocity near the goal
- **near_goal_angle:** Rewards correct heading at the goal
- **backwards_movement / lateral_movement:** Control movement direction preferences
- **SteppedProgressTerm:** Discrete reward for goal progress
- **AverageEpisodeVelocityTerm:** Rewards average velocity at goal

### Terminations & Events
- Goal-reaching conditions
- Timeouts based on goal distance
- Goal stability checks
- Reset events with terrain-aware spawn points
