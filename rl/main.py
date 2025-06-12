import datetime
import json
from pathlib import Path
from uuid import uuid4

import plant
from q_learning import QLearningEtaOptimizer

from utils import process_args


def main() -> None:
    args = process_args()
    patient_idx = args["patient_idx"]

    q_table_dir = Path.cwd() / "q_tables"
    q_table_dir.mkdir(exist_ok=True)

    breakpoints = [2, 200, 400, 600, 800, 1000]
    scenario_id = str(uuid4()).split("-")[0]

    initial_eta = 0.01
    optimizer = QLearningEtaOptimizer(initial_eta=initial_eta)
    episodes = 1_000

    for episode in range(episodes):
        metrics = plant.main(optimizer, patient_idx)
        print(f"Episode: {episode + 1} - Patient {patient_idx}")
        print(f"Reward: {optimizer.current_reward}")
        print(metrics)
        print(optimizer.q_table)
        print(optimizer.states_actions)
        print(optimizer.etas)
        print(f"Best actions/etas: {optimizer.get_best_actions_etas()}")
        print("\n\n")
        optimizer.reset_state_actions()
        optimizer.reset_eta(initial_eta)

        if episode + 1 in breakpoints:
            now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            output = {
                "q_table": optimizer.q_table.tolist(),
                "data": optimizer.get_best_actions_etas(),
            }
            q_table_file = q_table_dir / f"q_table_{scenario_id}_{now}.json"
            with Path.open(q_table_file, "w") as file:
                json.dump(output, file, indent=4, ensure_ascii=False)

    print("Q-table:")
    print(optimizer.q_table)
    print(f"Best actions/etas: {optimizer.get_best_actions_etas()}")

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output = {
        "q_table": optimizer.q_table.tolist(),
        "data": optimizer.get_best_actions_etas(),
    }
    q_table_file = q_table_dir / f"q_table_{scenario_id}_{now}.json"
    with Path.open(q_table_file, "w") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
