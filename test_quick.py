"""
Quick Test - Verify Environment & DQN Work
Run this first to check everything is set up correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch

print("="*60)
print("🧪 QUICK SYSTEM TEST")
print("="*60)

# Test 1: Imports
print("\n[1/5] Testing imports...")
try:
    from core.environment import PowerControlEnv
    from core.agent import DQNAgent
    from core.network import MassiveMIMOChannel
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: CUDA availability
print("\n[2/5] Checking CUDA...")
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️  CUDA not available, using CPU")
    device = "cpu"

# Test 3: Environment creation
print("\n[3/5] Creating environment...")
try:
    env = PowerControlEnv(
        network_config="config/network.yaml",
        dqn_config="config/dqn.yaml",
        user_idx=0
    )
    state = env.reset()
    print(f"✅ Environment created")
    print(f"   State dim: {env.get_state_dim()}")
    print(f"   Action dim: {env.get_action_dim()}")
    print(f"   Initial state: {state}")
except Exception as e:
    print(f"❌ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Agent creation
print("\n[4/5] Creating DQN agent...")
try:
    agent = DQNAgent(
        user_id=0,
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        config_path="config/dqn.yaml",
        device=device
    )
    print(f"✅ Agent created")
    print(f"   Epsilon: {agent.get_epsilon()}")
    print(f"   Device: {agent.trainer.device}")
except Exception as e:
    print(f"❌ Agent creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: One episode simulation
print("\n[5/5] Running test episode...")
try:
    state = env.reset()
    episode_reward = 0
    step_count = 0
    
    for _ in range(10):  # Just 10 steps
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        loss = agent.learn(state, action, reward, next_state, done)
        
        episode_reward += reward
        step_count += 1
        state = next_state
        
        if done:
            break
    
    print(f"✅ Test episode completed")
    print(f"   Steps: {step_count}")
    print(f"   Reward: {episode_reward:.3f}")
    print(f"   Final SINR: {info['sinr_dB']:.2f} dB")
    print(f"   Final EE: {info['energy_efficiency']:.2e} bit/J")
    
except Exception as e:
    print(f"❌ Episode simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\n🚀 Ready to start training:")
print("   python scripts/train.py --episodes 600")
print("\n" + "="*60)
