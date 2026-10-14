pipeline {
    agent any

    environment {
        IMAGE_NAME = "rag-pdf-qa"
        DOCKERHUB_CREDENTIALS = "dockerhub-credentials"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build and Test') {
            steps {
                sh '''
                    python3 --version
                    python3 -m py_compile app.py backend.py
                    test -f requirements.txt
                    test -f Dockerfile
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .'
                sh 'docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${IMAGE_NAME}:latest'
            }
        }

        stage('Push Image') {
            steps {
                echo 'Registry push will be configured after Docker Hub credentials are added.'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo 'Kubernetes deployment will be configured after Minikube is ready.'
            }
        }
    }

    post {
        always {
            echo 'CCA 2 pipeline execution finished.'
        }
    }
}
