//
// Created by uv on 10/6/20.
//

#include "data_training.h"

DataTraining::DataTraining()
{
    this->svm = cv::ml::SVM::create();
}

void DataTraining::SvmTraining(cv::Mat &train_data, cv::Mat &train_data_label)
{
    this->svm->setType(cv::ml::SVM::C_SVC);
    this->svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
    //this->svm->setGamma(0.01);
    //this->svm->setC(10.0);
    this->svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,3000,FLT_EPSILON));

    cv::Ptr<cv::ml::TrainData> input_data = cv::ml::TrainData::create(train_data,cv::ml::ROW_SAMPLE,train_data_label);

    std::cout<<"start SVM training..."<<std::endl;
    this->svm->train(input_data);
    std::cout<<"SVM training successfully!"<<std::endl;
    this->svm->save("/home/uv/auto_pilot/vision/hog_svm/result/svm_training.xml");
}

void DataTraining::SvmDetector()
{
    cv::Mat support_vector = this->svm->getSupportVectors();
    cv::Mat alpha,sv_index;
    float rho = this->svm->getDecisionFunction(0,alpha,sv_index);
    alpha.convertTo(alpha,CV_32FC1);

    cv::Mat result = alpha * support_vector;

    FILE *fp = fopen("/home/uv/auto_pilot/vision/hog_svm/result/svm_result.txt","wb");
    for(int i = 0;i < 3780;++i)
    {
        fprintf(fp,"%f \n",result.at<float>(0,i));
    }
    fprintf(fp,"%f",rho);
    fclose(fp);

    std::cout<<"save svm_rusult.txt successfully!"<<std::endl;
}

