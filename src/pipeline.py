import sys

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from src import train, monitor
from src.features import load_raw
from src.validate import validate_data

# cache_policy=NO_CACHE: our tasks pass large tables to each other, so we switch
# off Prefect's result caching, which would otherwise try to fingerprint them.


@task(name="load raw data", retries=2, retry_delay_seconds=10, cache_policy=NO_CACHE)
def load_data_task():
    return load_raw()


@task(name="validate data", cache_policy=NO_CACHE)
def validate_task(app, bureau, prev):
    return validate_data(app, bureau, prev)   # raises an error if the data is broken


@task(name="build features and split", cache_policy=NO_CACHE)
def prepare_task(app, bureau, prev):
    return train.prepare(app, bureau, prev)


@task(name="train candidate model", cache_policy=NO_CACHE)
def fit_task(data):
    return train.fit(data)


@task(name="evaluate and run gate", cache_policy=NO_CACHE)
def gate_task(model, data):
    result = train.evaluate_and_decide(model, data)
    train.print_report(result)
    return result


@task(name="save and promote or block", cache_policy=NO_CACHE)
def promote_task(model, data, result):
    return train.save_and_promote(model, data, result)


@task(name="record run in MLflow", cache_policy=NO_CACHE)
def record_task(data, result, decision):
    train.record_run(data, result, decision)


@flow(name="credit-risk-training", log_prints=True)
def training_flow():
    app, bureau, prev = load_data_task()

    for warning in validate_task(app, bureau, prev):
        print("WARNING:", warning)

    data = prepare_task(app, bureau, prev)
    model = fit_task(data)
    result = gate_task(model, data)
    decision = promote_task(model, data, result)
    record_task(data, result, decision)

    print(f"Pipeline finished. Gate decision: {decision}")
    return decision

@task(name="check data drift", cache_policy=NO_CACHE)
def drift_task(kind):
    return monitor.check_batch(kind)


@flow(name="credit-risk-monitoring", log_prints=True)
def monitoring_flow(batch: str = "no_drift"):
    report, retrain, reasons = drift_task(batch)
    print(report.to_string(index=False))

    if retrain:
        print("Drift detected: " + "; ".join(reasons))
        print("Starting the retraining pipeline...")
        return training_flow()          # runs as a sub-flow

    print("No significant drift. Keeping the current model.")
    return "no_action"

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    if command == "serve":
        training_flow.serve(name="weekly-retraining", cron="0 2 * * 0")
    elif command == "monitor":
        monitoring_flow(sys.argv[2] if len(sys.argv) > 2 else "no_drift")
    else:
        training_flow()