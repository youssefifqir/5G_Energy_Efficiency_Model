"""
Neo4j Graph Database Logger for 5G Power Control Visualization
Captures real-time network state, DQN decisions, and training metrics
"""

import yaml
import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
import numpy as np
from datetime import datetime

class Neo4jLogger:
    """
    Logs 5G power control simulation data to Neo4j graph database
    for dynamic visualization and analysis
    """

    _instance = None

    def __new__(cls, config_path="config/neo4j.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config/neo4j.yaml"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        """
        Initialize Neo4j connection and setup schema

        Args:
            config_path: Path to Neo4j configuration file
        """
        self.config = self._load_config(config_path)
        self.driver = None
        self.current_episode = None
        self.batch_buffer = []
        self.batch_size = self.config.get('batch_size', 50)

        self._connect()
        self._setup_schema()
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict:
        """Load Neo4j configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('neo4j', {})
        except FileNotFoundError:
            # Default configuration
            return {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'password',  # User should set this
                'database': 'powercontrol',
                'batch_size': 50
            }

    def _connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.config['uri'],
                auth=(self.config['username'], self.config['password'])
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Connected to Neo4j database")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            print("Make sure Neo4j Desktop is running and credentials are correct")
            raise

    def _setup_schema(self):
        """Create graph schema and constraints"""
        with self.driver.session() as session:
            # Create constraints for unique IDs
            session.run("""
                CREATE CONSTRAINT user_id IF NOT EXISTS
                FOR (u:User) REQUIRE u.id IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT bs_id IF NOT EXISTS
                FOR (b:BaseStation) REQUIRE b.id IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT episode_id IF NOT EXISTS
                FOR (e:Episode) REQUIRE e.id IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT agent_id IF NOT EXISTS
                FOR (a:DQNAgent) REQUIRE a.id IS UNIQUE
            """)

            print("✅ Graph schema initialized")

    def clear_database(self):
        """Clear all data from the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database cleared")

    def _setup_logging(self):
        """Setup Python logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Neo4jLogger')

    # ===============================
    # EPISODE MANAGEMENT
    # ===============================

    def start_episode(self, episode_id: int, n_users: int = 10):
        """
        Start a new training episode

        Args:
            episode_id: Episode number
            n_users: Number of users in simulation
        """
        print(f"Starting episode {episode_id} with {n_users} users")
        self.current_episode = episode_id

        with self.driver.session() as session:
            # Create episode node
            session.run("""
                CREATE (e:Episode {
                    id: $episode_id,
                    start_time: datetime(),
                    n_users: $n_users,
                    status: 'running'
                })
            """, episode_id=episode_id, n_users=n_users)

            # Create user nodes for this episode
            for user_id in range(n_users):
                session.run("""
                    MATCH (e:Episode {id: $episode_id})
                    CREATE (u:User {
                        id: $user_id,
                        episode_id: $episode_id,
                        power_level: 0.0,
                        sinr: 0.0,
                        qos_violated: false,
                        battery_level: 1.0,
                        channel_quality: 1.0,
                        self_interference: 0.0,
                        avg_power_3steps: 0.0
                    })-[:PARTICIPATES_IN]->(e)
                """, episode_id=episode_id, user_id=user_id)

            # Create base station
            session.run("""
                MATCH (e:Episode {id: $episode_id})
                CREATE (b:BaseStation {
                    id: 0,
                    episode_id: $episode_id,
                    n_antennas: 50,
                    bandwidth: 0.8,
                    noise_figure: 5.0
                })-[:SERVES]->(e)
            """, episode_id=episode_id)

            # Create DQN agents
            for user_id in range(n_users):
                session.run("""
                    MATCH (e:Episode {id: $episode_id})
                    CREATE (a:DQNAgent {
                        id: $user_id,
                        episode_id: $episode_id,
                        epsilon: 1.0,
                        total_steps: 0,
                        current_loss: 0.0
                    })-[:TRAINS_IN]->(e)
                """, episode_id=episode_id, user_id=user_id)

            self.logger.info(f"Started episode {episode_id} with {n_users} users")

    def end_episode(self, final_reward: float, final_ee: float, qos_violations: int):
        """
        End current episode with final metrics

        Args:
            final_reward: Final episode reward
            final_ee: Final energy efficiency
            qos_violations: Total QoS violations
        """
        if self.current_episode is None:
            return

        with self.driver.session() as session:
            session.run("""
                MATCH (e:Episode {id: $episode_id})
                SET e.end_time = datetime(),
                    e.status = 'completed',
                    e.final_reward = $final_reward,
                    e.final_ee = $final_ee,
                    e.qos_violations = $qos_violations
            """, episode_id=self.current_episode,
               final_reward=final_reward,
               final_ee=final_ee,
               qos_violations=qos_violations)

        self.logger.info(f"Ended episode {self.current_episode}")
        self.current_episode = None
        self._flush_batch()  # Ensure all data is saved

    # ===============================
    # STATE LOGGING
    # ===============================

    def log_user_state(self, user_id: int, state: np.ndarray, step: int):
        """
        Log user state vector

        Args:
            user_id: User identifier
            state: State vector [sinr, power, interference, qos_flag, battery, channel_quality, self_interference, avg_power]
            step: Current step in episode
        """
        if len(state) != 8:
            self.logger.warning(f"Invalid state length: {len(state)}, expected 8")
            return

        state_data = {
            'sinr': float(state[0]),
            'power': float(state[1]),
            'interference': float(state[2]),
            'qos_violated': bool(state[3]),
            'battery_level': float(state[4]),
            'channel_quality': float(state[5]),
            'self_interference': float(state[6]),
            'avg_power_3steps': float(state[7])
        }

        # Add to batch buffer
        self.batch_buffer.append({
            'type': 'user_state',
            'user_id': user_id,
            'step': step,
            'data': state_data
        })

        # Flush if batch is full
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()

    def log_agent_decision(self, user_id: int, state: np.ndarray, action: int,
                          q_values: np.ndarray, epsilon: float, reward: float, step: int):
        """
        Log DQN agent decision-making process

        Args:
            user_id: Agent/user identifier
            state: Current state
            action: Selected action
            q_values: Q-values for all actions
            epsilon: Current epsilon value
            reward: Reward received
            step: Current step
        """
        decision_data = {
            'user_id': user_id,
            'action': int(action),
            'q_values': q_values if isinstance(q_values, list) else (q_values.tolist() if q_values is not None else []),
            'epsilon': float(epsilon),
            'reward': float(reward),
            'step': step,
            'state_summary': {
                'sinr': float(state[0]) if len(state) > 0 else 0,
                'power': float(state[1]) if len(state) > 1 else 0
            }
        }

        self.batch_buffer.append({
            'type': 'agent_decision',
            'data': decision_data
        })

        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()

    def log_network_state(self, step: int, total_interference: float,
                         active_users: int, avg_sinr: float):
        """
        Log overall network state

        Args:
            step: Current step
            total_interference: Total network interference
            active_users: Number of active users
            avg_sinr: Average SINR across network
        """
        network_data = {
            'step': step,
            'total_interference': float(total_interference),
            'active_users': int(active_users),
            'avg_sinr': float(avg_sinr),
            'timestamp': datetime.now().isoformat()
        }

        self.batch_buffer.append({
            'type': 'network_state',
            'data': network_data
        })

    # ===============================
    # BATCH PROCESSING
    # ===============================

    def _flush_batch(self):
        """Flush accumulated data to Neo4j"""
        if not self.batch_buffer:
            return

        print(f"Flushing {len(self.batch_buffer)} items to Neo4j")

        with self.driver.session() as session:
            for item in self.batch_buffer:
                try:
                    if item['type'] == 'user_state':
                        self._save_user_state(session, item)
                    elif item['type'] == 'agent_decision':
                        self._save_agent_decision(session, item)
                    elif item['type'] == 'network_state':
                        self._save_network_state(session, item)
                except Exception as e:
                    print(f"Error saving {item['type']}: {e}")
                    self.logger.error(f"Error saving {item['type']}: {e}")

        self.batch_buffer.clear()

    def _save_user_state(self, session, item: Dict):
        """Save user state to Neo4j"""
        data = item['data']
        session.run("""
            MATCH (u:User {id: $user_id, episode_id: $episode_id})
            SET u.sinr = $sinr,
                u.power_level = $power,
                u.interference = $interference,
                u.qos_violated = $qos_violated,
                u.battery_level = $battery_level,
                u.channel_quality = $channel_quality,
                u.self_interference = $self_interference,
                u.avg_power_3steps = $avg_power_3steps,
                u.last_update = datetime()
        """, user_id=item['user_id'],
           episode_id=self.current_episode,
           **data)

    def _save_agent_decision(self, session, item: Dict):
        """Save agent decision to Neo4j"""
        data = item['data']
        session.run("""
            MATCH (a:DQNAgent {id: $user_id, episode_id: $episode_id})
            SET a.epsilon = $epsilon,
                a.last_action = $action,
                a.last_reward = $reward,
                a.total_steps = a.total_steps + 1,
                a.last_update = datetime()
            WITH a
            CREATE (d:Decision {
                step: $step,
                action: $action,
                q_values: $q_values,
                reward: $reward,
                sinr: $sinr,
                power: $power,
                timestamp: datetime()
            })-[:MADE_BY]->(a)
        """, user_id=data['user_id'],
           episode_id=self.current_episode,
           step=data['step'],
           action=data['action'],
           q_values=data['q_values'],
           reward=data['reward'],
           epsilon=data['epsilon'],
           sinr=data['state_summary']['sinr'],
           power=data['state_summary']['power'])

    def _save_network_state(self, session, item: Dict):
        """Save network state to Neo4j"""
        data = item['data']
        session.run("""
            MATCH (e:Episode {id: $episode_id})
            CREATE (n:NetworkState {
                step: $step,
                total_interference: $total_interference,
                active_users: $active_users,
                avg_sinr: $avg_sinr,
                timestamp: datetime()
            })-[:RECORDED_IN]->(e)
        """, episode_id=self.current_episode, **data)

    # ===============================
    # QUERY METHODS
    # ===============================

    def get_episode_summary(self, episode_id: int) -> Dict:
        """Get summary statistics for an episode"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Episode {id: $episode_id})
                RETURN e.final_reward as reward,
                       e.final_ee as ee,
                       e.qos_violations as violations,
                       e.n_users as users
            """, episode_id=episode_id)

            record = result.single()
            return dict(record) if record else {}

    def get_user_trajectory(self, episode_id: int, user_id: int) -> List[Dict]:
        """Get state trajectory for a specific user"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id, episode_id: $episode_id})
                RETURN u.sinr, u.power_level, u.qos_violated
                ORDER BY u.last_update
            """, episode_id=episode_id, user_id=user_id)

            return [dict(record) for record in result]

    def get_database_stats(self):
        """Get database statistics"""
        with self.driver.session() as session:
            # Count nodes by label
            labels = {}
            for label in ['Episode', 'User', 'BaseStation', 'DQNAgent', 'Decision', 'NetworkState']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()['count']
                labels[label] = count
            
            # Count relationships by type
            relationships = {}
            for rel_type in ['PARTICIPATES_IN', 'SERVES', 'TRAINS_IN', 'MADE_BY', 'RECORDED_IN']:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                count = result.single()['count']
                relationships[rel_type] = count
            
            return {
                'nodes': labels,
                'relationships': relationships
            }

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()