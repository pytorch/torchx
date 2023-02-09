

import unittest

from torchx.schedulers.azure_batch_scheduler import (
    AzureBatchOpts,
    AzureBatchScheduler,
    create_scheduler,
)

import torchx.schedulers.test.test_data as test_data

class AzureBatchSchedulerTest(unittest.TestCase):

    def setUp(self) -> None:
        self._scheduler = create_scheduler("test")
        self._pool_id = "testpool"

    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("session")
        self.assertIsInstance(scheduler, AzureBatchScheduler)


    def test_build_client(self) -> None:
        scheduler = create_scheduler(
            "session",
            batch_account_name="account_name",
            batch_account_key="account_key",
            batch_account_url="account_url",
        )

        batch_client = scheduler._build_client()
        self.assertEqual("account_name", 
            batch_client.config.credentials.auth._account_name)
        self.assertEqual("account_key", 
            batch_client.config.credentials.auth._key)
        self.assertEqual("account_url", 
            batch_client.config.batch_url)
        

    def test_submit_dryrun_with_pool_id_set_request_pool_id(self) -> None:
        app = test_data.build_app_def()
        cfg = AzureBatchOpts({"batch_pool_id": self._pool_id})
        info = self._scheduler.submit_dryrun(app, cfg)

        req = info.request
        
        self.assertEqual(self._pool_id, req.pool_id)

    def test_submit_dryrun_with_pool_id_set_job_pool_id(self) -> None:
        app = test_data.build_app_def()
        cfg = AzureBatchOpts({"batch_pool_id": self._pool_id})
        info = self._scheduler.submit_dryrun(app, cfg)

        req = info.request
        
        self.assertEqual(self._pool_id, req.job.pool_info.pool_id)


    def test_submit_dryrun_job(self) -> None:
        app = test_data.build_app_def()
        app.metadata = {"prop1":"val1"}
        cfg = AzureBatchOpts({"batch_pool_id": self._pool_id})

        info = self._scheduler.submit_dryrun(app, cfg)
        req = info.request        
        job = req.job

        self.assertTrue(job.id.startswith(app.name))
        self.assertEqual(job.id, job.display_name)

        self.assertEqual(job.id, job.display_name)

        self.assertEqual(job.pool_info.pool_id, self._pool_id)
        self.assertEqual(len(job.metadata), 1)

    def test_submit_dryrun_tasks(self) -> None:
        app = test_data.build_app_def()
        cfg = AzureBatchOpts({"batch_pool_id": self._pool_id})

        info = self._scheduler.submit_dryrun(app, cfg)
        req = info.request        
        tasks = req.tasks
        trainer_task = tasks[0]

        self.assertEqual(len(tasks), 1)

        env = {prop.name:prop.value for prop in trainer_task.environment_settings}
        self.assertEqual(env["FOO"], app.roles[0].env["FOO"])
        self.assertEqual(env["TORCHX_ROLE_NAME"], app.roles[0].name)

        self.assertEqual(trainer_task.multi_instance_settings.number_of_instances,
            app.roles[0].num_replicas)

        self.assertEqual(trainer_task.multi_instance_settings.coordination_command_line,
            f"main --output-path  --app-id {req.job.id} --rank0_env AZ_BATCH_MASTER_NODE")
