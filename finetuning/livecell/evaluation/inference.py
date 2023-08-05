import argparse
import os
from subprocess import run

import micro_sam.evaluation as evaluation
from micro_sam.evaluation.livecell import livecell_inference
from util import check_model, get_checkpoint, get_experiment_folder, DATA_ROOT, PROMPT_FOLDER


def inference_job(prompt_settings, model_name):
    experiment_folder = get_experiment_folder(model_name)
    checkpoint, model_type = get_checkpoint(model_name)
    livecell_inference(
        checkpoint,
        input_folder=DATA_ROOT,
        model_type=model_type,
        experiment_folder=experiment_folder,
        use_points=prompt_settings["use_points"],
        use_boxes=prompt_settings["use_boxes"],
        n_positives=prompt_settings["n_positives"],
        n_negatives=prompt_settings["n_negatives"],
        prompt_folder=PROMPT_FOLDER,
    )


def submit_array_job(prompt_settings, model_name, test_run):
    n_settings = len(prompt_settings)
    cmd = ["sbatch", "-a", f"0-{n_settings-1}", "inference.sbatch", "-n", model_name]
    if test_run:
        cmd.append("-t")
    run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-t", "--test_run", action="store_true")
    args = parser.parse_args()

    if args.test_run:  # test run with only the three default experiment settings
        settings = evaluation.default_experiment_settings()
    else:  # all experiment settings
        settings = evaluation.full_experiment_settings()
        settings.extend(evaluation.full_experiment_settings(use_boxes=True))

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    model_name = args.name
    check_model(model_name)

    if job_id is None:  # this is the main script that submits slurm jobs
        submit_array_job(settings, model_name, args.test_run)
    else:  # we're in a slurm job and run inference for a setting
        job_id = int(job_id)
        this_settings = settings[job_id]
        inference_job(this_settings, model_name)


if __name__ == "__main__":
    main()
