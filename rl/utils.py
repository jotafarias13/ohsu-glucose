import argparse


def is_multiple(large: int, small: int) -> bool:
    return large % small == 0


def process_args() -> dict:
    parser = argparse.ArgumentParser(
        prog="Executa Q Table para o número do paciente",
    )

    parser.add_argument(
        "-p",
        "--patient",
        type=int,
        default=None,
        help="Número do paciente",
    )

    args = parser.parse_args()
    patient = args.patient

    return {"patient_idx": patient}
