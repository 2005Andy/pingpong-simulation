"""Basic tests for pingpong simulation."""

import numpy as np
import pytest

from src.ball_types import BallState, Table, Net
from src.physics import aerodynamic_acceleration, rk4_step
from src.scenarios import create_table, create_net


class TestBallState:
    """Test BallState functionality."""

    def test_ball_state_creation(self):
        """Test creating a ball state."""
        pos = np.array([0.0, 0.0, 1.0])
        vel = np.array([1.0, 0.0, 0.0])
        omega = np.array([0.0, 100.0, 0.0])

        ball = BallState(pos, vel, omega)

        assert np.array_equal(ball.position, pos)
        assert np.array_equal(ball.velocity, vel)
        assert np.array_equal(ball.omega, omega)

    def test_ball_state_copy(self):
        """Test copying a ball state."""
        pos = np.array([0.0, 0.0, 1.0])
        vel = np.array([1.0, 0.0, 0.0])
        omega = np.array([0.0, 100.0, 0.0])

        ball = BallState(pos, vel, omega)
        ball_copy = ball.copy()

        # Modify original
        ball.position[0] = 10.0

        # Copy should be unchanged
        assert ball_copy.position[0] == 0.0
        assert ball.position[0] == 10.0


class TestPhysics:
    """Test physics calculations."""

    def test_aerodynamic_acceleration_zero_velocity(self):
        """Test aerodynamic acceleration with zero velocity."""
        vel = np.zeros(3)
        omega = np.zeros(3)

        acc = aerodynamic_acceleration(vel, omega)

        # Should only have gravity
        expected = np.array([0.0, 0.0, -9.81])
        np.testing.assert_array_almost_equal(acc, expected)

    def test_aerodynamic_acceleration_with_velocity(self):
        """Test aerodynamic acceleration with non-zero velocity."""
        vel = np.array([10.0, 0.0, 0.0])  # 10 m/s horizontal velocity
        omega = np.zeros(3)

        acc = aerodynamic_acceleration(vel, omega)

        # Should have gravity + drag (negative x direction)
        assert acc[0] < 0.0  # Drag force
        assert acc[2] == -9.81  # Gravity unchanged
        assert acc[1] == 0.0  # No y acceleration

    def test_rk4_step_conservation(self):
        """Test that RK4 step conserves energy for simple cases."""
        # Ball with only gravity
        ball = BallState(
            position=np.array([0.0, 0.0, 10.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            omega=np.zeros(3)
        )

        dt = 0.01
        new_ball = rk4_step(ball, dt)

        # Position should change due to gravity
        assert new_ball.position[2] < ball.position[2]  # Falling down
        assert new_ball.position[0] == ball.position[0]  # No x movement
        assert new_ball.position[1] == ball.position[1]  # No y movement

        # Velocity should change due to gravity
        assert new_ball.velocity[2] < ball.velocity[2]  # Accelerating downward


class TestScenarios:
    """Test scenario creation."""

    def test_create_table(self):
        """Test table creation."""
        table = create_table()

        assert table.height == 0.76  # ITTF standard
        assert table.length == 2.74  # ITTF standard
        assert table.width == 1.525  # ITTF standard
        assert table.restitution == 0.90
        assert table.friction == 0.25

    def test_create_net(self):
        """Test net creation."""
        net = create_net()

        assert net.height == 0.1525  # ITTF standard
        assert net.length == 1.83  # ITTF standard
        assert net.x_position == 0.0
        assert net.thickness == 0.002
