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

# DEの実務スキル：セキュリティ（パブリックアクセスを完全に遮断）


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

# 5. アウトプット
output "main_bucket_id" {
  value = module.kaggle_s3.s3_bucket_id
}