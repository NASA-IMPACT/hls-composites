"""Exit codes the container returns, and the job monitor interprets.

Kept free of imports so both the application and the CDK code can read them:
an exit code is a contract between the container and the monitoring
infrastructure, and duplicating the number is how the two drift apart.
"""

NO_INPUTS = 5
"""No granules were found for the requested tile and period.

Not a fault: an early-mission month or a sparse high-latitude tile legitimately
has nothing to composite. The job monitor maps this to `FAILURE_NO_INPUTS`.
"""
