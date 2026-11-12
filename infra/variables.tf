variable "aws_region" {
    description = "AWS region where MonoVideo3D resources are created."
    type = string
    default = "us-east-1"
}

variable "environment" {
    description = "Deployment environment used in resource names and tags; must be dev or prod."
    type = string
    default = "dev"

    validation {
        condition = contains(["dev", "prod"], var.environment)
        error_message = "Environment must be one of: dev, prod."
    }
}
