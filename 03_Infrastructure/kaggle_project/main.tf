# 1. AWSプロバイダーの設定
provider "aws" {
  region = "ap-northeast-1" 
}

# 2. メインのデータ用バケット
module "kaggle_s3" {
  source      = "../../AWS_IaC_Terraform/modules/s3_bucket" 
  bucket_name = "kosato-heart-disease-v9-data" 
  environment = "dev"
  tags        = { Project = "HeartDisease", Owner = "Sato" } # ここに追加
}

# 3. 学習用データ保管バケット
module "s3_training_data" {
  source      = "../../AWS_IaC_Terraform/modules/s3_bucket"
  bucket_name = "sato-ds-project-training-data-2026"
  environment = "dev"
  tags        = { Project = "HeartDisease", Owner = "Sato" } # ここにも追加
}

# 4. アウトプット
output "main_bucket_id" {
  value = module.kaggle_s3.s3_bucket_id
}