pipeline {
    agent any

    environment {
        PYTHONPATH = 'src'
        IMAGE_NAME = 'voyage-flight-price-api'
    }

    stages {
        stage('Install') {
            steps {
                sh 'python3 -m pip install --upgrade pip'
                sh 'python3 -m pip install -r requirements.txt'
            }
        }

        stage('Test') {
            steps {
                sh 'pytest -q'
            }
        }

        stage('Train') {
            steps {
                sh 'python3 -m voyage_ml.train_regression'
                sh 'python3 -m voyage_ml.train_gender_classifier'
                sh 'python3 -m voyage_ml.recommender'
            }
        }

        stage('Build Image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .'
                sh 'docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${IMAGE_NAME}:latest'
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'kubectl apply -f k8s/'
                sh 'kubectl rollout status deployment/voyage-flight-price-api'
            }
        }
    }
}
