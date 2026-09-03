from typing import Any, cast

from aws_cdk import (
    CfnOutput,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_iam as iam,
    custom_resources as cr,
)
from constructs import Construct


class BatchInfra(Construct):
    """AWS Batch compute environment and job queue."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        instance_classes: list[str] | None,
        max_vcpu: int,
        ami_id: str,
        job_queue_name: str,
        stage: str,
        **kwargs: Any,
    ) -> None:
        """Set up an AWS Batch ComputeEnvironment and JobQueue.

        Parameters
        ----------
        vpc:
            VPC in which the ComputeEnvironment will launch instances.
        instance_classes:
            If provided, limit the ComputeEnvironment to these instance types.
            Instance types are given as strings and converted into the matching
            `ec2.InstanceClass` enum. If empty, AWS Batch uses "optimal" instance
            classes.
        max_vcpu:
            Maximum number of vCPUs in the ComputeEnvironment.
        ami_id:
            AWS Batch EC2 instance AMI identifier, OR the name of an SSM parameter
            holding the AMI ID prefixed by `resolve:ssm:` (e.g.
            `resolve:ssm:/param-name`).
        job_queue_name:
            Name of the JobQueue jobs are submitted to.
        stage:
            Deployment stage, carried into the ComputeEnvironment's construct id.
            A replaceable compute environment cannot be given an explicit name,
            and its generated name is the first 24 characters of its logical id
            plus a hash -- so the stage has to appear early in the id to be
            readable at all.
        """
        super().__init__(scope, construct_id, **kwargs)

        if instance_classes:
            ec2_instance_classes = [
                ec2.InstanceClass(instance_class) for instance_class in instance_classes
            ]
        else:
            ec2_instance_classes = None

        # AWS Batch appends its own sections to the instance UserData, which it can
        # only do if the UserData we supply is multi-part.
        multipart_user_data = ec2.MultipartUserData(parts_separator="==BOUNDARY==")
        command_user_data = ec2.UserData.for_linux()
        command_user_data.add_commands(
            # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/pull-behavior.html
            'echo "ECS_IMAGE_PULL_BEHAVIOR=prefer-cached" >> /etc/ecs/ecs.config',
        )
        multipart_user_data.add_part(
            ec2.MultipartBody.from_user_data(command_user_data)
        )

        if ami_id.startswith("resolve:ssm:"):
            # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-launch-template.html#use-an-ssm-parameter-instead-of-an-ami-id
            ec2_machine_image = ec2.MachineImage.resolve_ssm_parameter_at_launch(
                ami_id.removeprefix("resolve:ssm:")
            )
        else:
            ec2_machine_image = ec2.MachineImage.lookup(name=ami_id)
        ecs_machine_image = batch.EcsMachineImage(
            image=ec2_machine_image,
            image_type=batch.EcsMachineImageType.ECS_AL2023,
        )
        launch_template = ec2.LaunchTemplate(
            self,
            "LaunchTemplate",
            machine_image=ec2_machine_image,
            user_data=multipart_user_data,
        )

        # SSM read access lets the instances resolve the AMI parameter at launch.
        compute_environment_service_role = iam.Role(
            self,
            "ServiceRole",
            assumed_by=iam.ServicePrincipal("batch.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSBatchServiceRole"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMReadOnlyAccess"
                ),
            ],
        )

        self.compute_environment = batch.ManagedEc2EcsComputeEnvironment(
            self,
            f"CE-{stage.capitalize()}",
            allocation_strategy=batch.AllocationStrategy.SPOT_CAPACITY_OPTIMIZED,
            images=[ecs_machine_image],
            launch_template=launch_template,
            instance_classes=ec2_instance_classes,
            use_optimal_instance_classes=ec2_instance_classes is None,
            spot=True,
            minv_cpus=0,
            maxv_cpus=max_vcpu,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
            ),
            vpc=vpc,
            terminate_on_update=False,
            service_role=compute_environment_service_role,
            # Replacing the compute environment allows updating more of its settings,
            # at the cost of not being able to give it a stable name.
            replace_compute_environment=True,
        )

        # ManagedEc2EcsComputeEnvironment requires an override to track `$Latest`
        # Ref: https://github.com/aws/aws-cdk/issues/28137
        cfn_ce = cast(
            batch.CfnComputeEnvironment,
            self.compute_environment.node.find_child("Resource"),
        )
        cfn_ce.add_property_override(
            "ComputeResources.LaunchTemplate.Version",
            launch_template.latest_version_number,
        )
        self.enable_container_insights(self.compute_environment)

        self.queue = batch.JobQueue(
            self,
            "JobQueue",
            job_queue_name=job_queue_name,
        )
        self.queue.add_compute_environment(self.compute_environment, 1)

        CfnOutput(
            self,
            "JobQueueName",
            value=self.queue.job_queue_name,
        )

    def enable_container_insights(
        self, compute_environment: batch.IComputeEnvironment
    ) -> None:
        """Enable ContainerInsights for this managed ComputeEnvironment.

        AWS Batch owns the underlying ECS cluster, so the setting has to be applied
        out of band once the cluster ARN is known.
        Ref: https://github.com/aws/aws-cdk/issues/21698#issuecomment-1898890043
        """
        batch_ecs_cluster = cr.AwsCustomResource(
            self,
            "BatchEcsCluster",
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
            ),
            on_update=cr.AwsSdkCall(
                service="@aws-sdk/client-batch",
                action="DescribeComputeEnvironmentsCommand",
                parameters={
                    "computeEnvironments": [
                        compute_environment.compute_environment_arn
                    ],
                },
                physical_resource_id=cr.PhysicalResourceId.from_response(
                    "computeEnvironments.0.ecsClusterArn"
                ),
            ),
        )
        cr.AwsCustomResource(
            self,
            "EnableContainerInsights",
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
            ),
            on_update=cr.AwsSdkCall(
                service="@aws-sdk/client-ecs",
                action="UpdateClusterCommand",
                parameters={
                    "cluster": batch_ecs_cluster.get_response_field_reference(
                        "computeEnvironments.0.ecsClusterArn"
                    ),
                    "settings": [
                        {
                            "name": "containerInsights",
                            "value": "enabled",
                        }
                    ],
                },
                physical_resource_id=cr.PhysicalResourceId.of("compute-resource-tags"),
            ),
        )
