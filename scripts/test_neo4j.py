#!/usr/bin/env python3
"""
Test script for Neo4j integration with 5G Power Control
Tests connection, schema creation, and basic data logging
"""

import sys
import os
import numpy as np

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from utils.neo4j_logger import Neo4jLogger

def test_connection():
    """Test basic Neo4j connection"""
    print("🔗 Testing Neo4j connection...")

    try:
        logger = Neo4jLogger()
        print("✅ Connection successful!")
        return logger
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure:")
        print("  1. Neo4j Desktop is running")
        print("  2. Password in config/neo4j.yaml is correct")
        print("  3. Database is started")
        return None

def test_episode_management(logger):
    """Test episode creation and management"""
    print("\n📊 Testing episode management...")

    # Start episode
    logger.start_episode(episode_id=999, n_users=3)
    print("✅ Episode 999 started")

    # Simulate some user states
    for user_id in range(3):
        # Create fake state vector [sinr, power, interference, qos_flag, battery, channel_quality, self_interference, avg_power]
        state = np.array([
            np.random.uniform(5, 20),  # SINR
            np.random.uniform(-50, -10),  # Power (dBW)
            np.random.uniform(0, 10),  # Interference
            np.random.choice([0, 1]),  # QoS violated
            np.random.uniform(0.5, 1.0),  # Battery
            np.random.uniform(0.8, 1.2),  # Channel quality
            np.random.uniform(0, 5),  # Self interference
            np.random.uniform(-50, -10)  # Avg power 3 steps
        ])

        logger.log_user_state(user_id=user_id, state=state, step=1)

    print("✅ User states logged")

    # End episode
    logger.end_episode(final_reward=150.5, final_ee=250000, qos_violations=2)
    print("✅ Episode 999 completed")

def test_agent_decisions(logger):
    """Test agent decision logging"""
    print("\n🤖 Testing agent decision logging...")

    # Start a new episode for testing
    logger.start_episode(episode_id=1000, n_users=2)

    for step in range(5):
        for user_id in range(2):
            # Fake state
            state = np.random.rand(8)

            # Fake Q-values
            q_values = np.random.rand(16)  # 16 actions

            # Fake decision
            action = np.random.randint(0, 16)
            reward = np.random.uniform(-10, 10)
            epsilon = 0.1

            logger.log_agent_decision(
                user_id=user_id,
                state=state,
                action=action,
                q_values=q_values,
                epsilon=epsilon,
                reward=reward,
                step=step
            )

    print("✅ Agent decisions logged")

    # End episode
    logger.end_episode(final_reward=200.0, final_ee=300000, qos_violations=0)

def test_queries(logger):
    """Test data retrieval queries"""
    print("\n🔍 Testing data queries...")

    # Get episode summary
    summary = logger.get_episode_summary(999)
    if summary:
        print(f"✅ Episode 999 summary: {summary}")
    else:
        print("⚠️ No summary found for episode 999")

    # Get user trajectory
    trajectory = logger.get_user_trajectory(999, 0)
    if trajectory:
        print(f"✅ User 0 trajectory: {len(trajectory)} data points")
    else:
        print("⚠️ No trajectory found for user 0")

def main():
    """Main test function"""
    print("🧪 Neo4j Integration Test for 5G Power Control")
    print("=" * 50)

    # Test connection
    logger = test_connection()
    if not logger:
        return

    try:
        # Test episode management
        test_episode_management(logger)

        # Test agent decisions
        test_agent_decisions(logger)

        # Test queries
        test_queries(logger)

        print("\n🎉 All tests completed successfully!")
        print("\n📋 Next steps:")
        print("  1. Open Neo4j Browser: http://localhost:7474")
        print("  2. Run: MATCH (n) RETURN n LIMIT 25")
        print("  3. Explore your 5G network graph!")
        print("  4. Integrate with your training scripts")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if logger:
            logger.close()

if __name__ == "__main__":
    main()
</content>
<parameter name="filePath">c:\Users\youss\Downloads\power-control-dqn-FINAL\power-control-dqn\scripts\test_neo4j.py