# 1. AWSプロバイダーの設定
provider "aws" {
  region = "ap-northeast-1"
}

# 2. 変数の定義
variable "project_tag" {
  default = "HeartDisease"
}

# 3. メインのデータ用バケット
module "kaggle_s3" {
  source      = "../../AWS_IaC_Terraform/modules/s3_bucket"
  bucket_name = "kosato-heart-disease-v9-data"
  environment = "dev"
  tags        = { Project = var.project_tag, Owner = "Sato" }
}

# 4. 学習用データ保管バケット
module "s3_training_data" {
  source      = "../../AWS_IaC_Terraform/modules/s3_bucket"
  bucket_name = "sato-ds-project-training-data-2026"
  environment = "dev"
  tags        = { Project = var.project_tag, Owner = "Sato" }

  lifecycle_rule = [
    {
      id      = "auto-delete-old-data"
      enabled = true
      expiration = {
        days = 30
      }
    }
  ]
}

# 5. セキュリティ（パブリックアクセスブロック）
resource "aws_s3_bucket_public_access_block" "kaggle_s3_block" {
  bucket = module.kaggle_s3.s3_bucket_id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "training_data_block" {
  bucket = module.s3_training_data.s3_bucket_id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 6. IAM RoleとPolicy（DEの実務スキル）

resource "aws_iam_policy" "s3_access_policy" {
  name        = "S3AccessPolicy"
  description = "Allow read and write access to specific S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Effect   = "Allow"
        Resource = [
          module.kaggle_s3.s3_bucket_arn,
          "${module.kaggle_s3.s3_bucket_arn}/*",
          module.s3_training_data.s3_bucket_arn,
          "${module.s3_training_data.s3_bucket_arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role" "data_pipeline_role" {
  name = "DataPipelineRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "s3_access_attach" {
  role       = aws_iam_role.data_pipeline_role.name
  policy_arn = aws_iam_policy.s3_access_policy.arn
}

# IAM Instance Profile (RoleをEC2に紐付けるための「パスケース」)
resource "aws_iam_instance_profile" "pipeline_profile" {
  name = "data-pipeline-instance-profile"
  # role名を上で定義した名前に合わせる
  role = aws_iam_role.data_pipeline_role.name

  tags = {
    Project = var.project_tag
  }
}