output "dynamodb_table_name" {
    description = "Name of the on-demand DynamoDB jobs table."
    value       = aws_dynamodb_table.jobs.name
}
