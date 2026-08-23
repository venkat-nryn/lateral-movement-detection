# Threat Model

This document describes the threat scenario the project is designed to defend against. The project focuses exclusively on defensive detection and analysis; no offensive tooling or automation is developed.

## Adversary Behaviour Assumed

1. **Initial access**: The attacker gains an initial foothold on a host inside the enterprise network (e.g., via phishing, exploitation of an exposed service, or supply-chain compromise).

2. **Use of legitimate credentials**: After initial access, the attacker may use legitimate (stolen or forged) credentials. This makes the attacker's activity look similar to ordinary user behaviour, complicating detection.

3. **Lateral movement across hosts**: The attacker moves from the initially compromised host to other hosts in the network, often in multiple hops, using protocols and credentials available in the environment.

4. **Targeting privileged / critical assets**: The attacker's ultimate goal is assumed to be reaching privileged accounts, domain infrastructure, or other critical assets (e.g., domain controllers, sensitive data stores).

## Defensive Focus

- The project's goal is **defensive detection**: identifying lateral-movement activity in security logs and reconstructing likely attack paths so analysts can respond.
- Outputs are intended to support human analysts with detections, confidence/uncertainty estimates, risk scores, and explanations.
- All modelling assumes access to benign enterprise telemetry plus (for evaluation purposes) labelled or simulated attack scenarios.

## Out of Scope

- **Automatic offensive actions** of any kind are out of scope: no exploit generation, credential theft, automated attack execution, or autonomous response that modifies systems.
- Any automated action beyond producing analytical output for analysts is explicitly not part of this project.
