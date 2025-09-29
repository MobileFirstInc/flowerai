Flower Dashboard
================

An interactive Streamlit dashboard that visualises metrics from a Flower federated learning experiment. The app embraces a plug-and-play workflow: drop in a metrics export and immediately inspect client participation, model convergence, and client-level anomalies.

Features
--------

* 📊 Line charts for loss and accuracy trends across rounds.
* 🙋 Client participation breakdown that highlights which clients finished, straggled, or dropped.
* 🧮 Round-by-round contribution analysis using example counts (or aggregation weights when available).
* 🚨 Automated alerts for stragglers, dropped clients, and anomalous client updates (z-score based).
* ⚙️ Configurable detection thresholds directly from the sidebar.

Getting started
---------------

1. Install dependencies::

      pip install -r requirements.txt

2. Launch the dashboard::

      streamlit run streamlit_app.py

3. Load metrics

   * Toggle the "Use bundled sample data" switch to explore the included ``sample_metrics.json`` in ``assets``.
   * Or upload your own metrics export (Flower simulation history, aggregator callback logs, or a custom JSON that mirrors the sample schema).

Expected JSON schema
--------------------

The dashboard expects a JSON document with a list of rounds. Every round entry may contain global metrics (loss, accuracy, server time) and a list of ``clients`` with per-client metrics.

.. code-block:: json

   {
     "rounds": [
       {
         "round": 1,
         "loss": 1.85,
         "accuracy": 0.45,
         "server_time": 4.2,
         "clients": [
           {
             "client_id": "client_1",
             "status": "completed",
             "loss": 1.95,
             "accuracy": 0.40,
             "examples": 128,
             "duration": 12.1
           }
         ]
       }
     ]
   }

Generating metrics from Flower code
-----------------------------------

There are several ways to produce a compatible JSON file:

* Use ``History.to_json()`` from a Flower simulation and save the result to disk.
* Collect metrics in a strategy callback (for example ``evaluate_round``) and export them when training finishes.
* Record client-side metrics (examples, loss, duration) in the ``fit`` return payload. The dashboard will automatically switch to aggregation weights if ``examples`` are missing.

Extending
---------

* Replace the static JSON loader with a websocket, database, or message-bus consumer to power near-real-time monitoring.
* Add authentication and deploy the dashboard to a shared monitoring cluster via Streamlit Community Cloud or Flower Intelligence.

License
-------

This example follows the main project license. See :code:`../../LICENSE`.
