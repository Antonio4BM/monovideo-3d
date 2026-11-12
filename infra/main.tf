terraform {
    required_providers {
        aws = {
            source = "hashicorp/aws"
            version = "~> 6.0"
        }
    }
}


provider "aws" {
    region = var.aws_region

    default_tags {
        tags = {
            Project = "monovideo3d"
            Environment = var.environment
            ManagedBy = "terraform"
        }
    }
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "monovideo" {
    bucket = "monovideo3d-${var.environment}-${data.aws_caller_identity.current.account_id}"

    force_destroy = true

    lifecycle {
        prevent_destroy = false
    }
}

resource "aws_s3_bucket_public_access_block" "monovideo" {
    bucket = aws_s3_bucket.monovideo.id

    # blocks public access to the bucket and its objects with old ACLs and policies
    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "monovideo" {
    bucket = aws_s3_bucket.monovideo.id

    # defines ownership controls for the bucket, ensuring the bucket owner has full control over the objects in the bucket
    rule {
        object_ownership = "BucketOwnerPreferred"
    }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "monovideo" {
    bucket = aws_s3_bucket.monovideo.id

    # defines the server side encryption configuration
    rule {
        apply_server_side_encryption_by_default {
            sse_algorithm = "AES256"
        }
        # blocks the use of customer provided encryption keys
        blocked_encryption_types = ["SSE-C"]
    }
}

resource "aws_s3_bucket_versioning" "monovideo" {
    bucket = aws_s3_bucket.monovideo.id

    versioning_configuration {
        status = "Enabled"
    }
}

resource "aws_s3_bucket_lifecycle_configuration" "monovideo" {
    bucket = aws_s3_bucket.monovideo.id

    rule {
        id = "remove-incomplete-uploads"
        status = "Enabled"

        filter {}

        abort_incomplete_multipart_upload {
            days_after_initiation = 7
        }
    }

    rule {
        id     = "expire-uploads"
        status = "Enabled"

        filter {
            prefix = "uploads/"
        }

        expiration {
            days = 15
        }

        noncurrent_version_expiration {
            noncurrent_days = 1
        }
    }

    rule {
        id     = "remove-old-result-versions"
        status = "Enabled"

        filter {
            prefix = "results/"
        }

        noncurrent_version_expiration {
            noncurrent_days = 7
        }
    }

    rule {
        id     = "remove-expired-delete-markers"
        status = "Enabled"

        filter {}

        expiration {
            expired_object_delete_marker = true
        }
    }

    depends_on = [aws_s3_bucket_versioning.monovideo]
}

data "aws_iam_policy_document" "required_tls" {
    statement {
        sid = "DenyInsecureTransport"
        effect = "Deny"

        actions = ["s3:*"]

        resources = [
            aws_s3_bucket.monovideo.arn,
            "${aws_s3_bucket.monovideo.arn}/*",
        ]

        principals {
           type = "*"
           identifiers = ["*"] 
        }

        condition {
            test = "Bool"
            variable = "aws:SecureTransport"
            values = ["false"]
        }
    }
}

resource "aws_s3_bucket_policy" "required_tls" {
    bucket = aws_s3_bucket.monovideo.id
    policy = data.aws_iam_policy_document.required_tls.json

    depends_on = [aws_s3_bucket_ownership_controls.monovideo]
}
