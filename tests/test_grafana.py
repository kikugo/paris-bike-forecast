import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "grafana"


def test_alert_rules_use_the_provisioned_datasource():
    rules = yaml.safe_load((ROOT / "provisioning/alerting/rules.yml").read_text())
    uids = {r["uid"] for g in rules["groups"] for r in g["rules"]}
    assert uids == {"bikes-psi-drift", "bikes-low-coverage"}
    for group in rules["groups"]:
        for rule in group["rules"]:
            queries = [d for d in rule["data"] if d["datasourceUid"] != "__expr__"]
            assert [q["datasourceUid"] for q in queries] == ["bikes-pg"]
            assert rule["condition"] in {d["refId"] for d in rule["data"]}


def test_dashboards_parse_and_alerts_point_at_real_panels():
    dash = json.loads((ROOT / "dashboards/model_monitoring.json").read_text())
    panel_ids = {p["id"] for p in dash["panels"]}
    rules = yaml.safe_load((ROOT / "provisioning/alerting/rules.yml").read_text())
    for rule in rules["groups"][0]["rules"]:
        assert rule["dashboardUid"] == dash["uid"] and rule["panelId"] in panel_ids
