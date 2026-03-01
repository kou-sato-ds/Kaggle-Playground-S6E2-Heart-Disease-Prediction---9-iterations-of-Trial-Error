# --- 03_Infrastructure/kaggle_project/main.tf ---

# 1. AWSプロバイダーの設定
provider "aws" {
  region = "ap-northeast-1" # 東京リージョン
}

# 2. 部品庫（modules）からS3作成機能を呼び出す
module "kaggle_s3" {
  # パスとバケット名をプロ仕様に修正
  source      = "../../AWS_IaC_Terraform/modules/s3_bucket" 
  bucket_name = "kosato-heart-disease-v9-data" # アンダースコアをハイフンに統一
  environment = "dev"
}

# 3. アウトプット（作成されたバケットIDを画面に表示）
output "bucket_id" {
  value = module.kaggle_s3.s3_bucket_id
}