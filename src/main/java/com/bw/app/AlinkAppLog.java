package com.bw.app;

import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.classification.*;
import com.alibaba.alink.operator.batch.dataproc.SplitBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalBinaryClassBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalMultiClassBatchOp;
import com.alibaba.alink.operator.batch.sink.CsvSinkBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.operator.common.evaluation.BinaryClassMetrics;
import com.alibaba.alink.operator.common.evaluation.MultiClassMetrics;
import com.alibaba.alink.operator.stream.StreamOperator;

import static com.alibaba.alink.operator.batch.graph.KCoreBatchOp.KCore.k;

public class AlinkAppLog {
    public static void main(String[] args) throws Exception {
        //（1）、构建Alink开发环境，设置批处理并行度为1，加载数据集，控制台输出样本数据和条目数；（5分）
        BatchOperator.setParallelism(1);

        String filePath = "src/main/java/com/bw/bean/seeds_dataset.txt";
        String schema = "x1 double," +
                        "x2 double," +
                        "x3 double," +
                        "x4 double," +
                        "x5 double," +
                        "x6 double," +
                        "x7 double," +
                        "y int";
        CsvSourceBatchOp csvSource = new CsvSourceBatchOp()
                .setFilePath(filePath)
                .setSchemaStr(schema)
                .setFieldDelimiter("\t");
        //todo 样本数据
        csvSource.print(5);

        //todo 条目数
        System.out.println(csvSource.count());

        //、数据集划分：调用Alink库中API，按照8：2比例划分训练数据集为trainData和testData，并且输出条目数；（5分）
        BatchOperator <?> spliter = new SplitBatchOp().setFraction(0.8);
        BatchOperator<?> trainData = spliter.linkFrom(csvSource);
        BatchOperator<?> testData = spliter.getSideOutput(0);

        //todo 输出条目数
        System.out.println(trainData.count());
        System.out.println(testData.count());


        //（5）、模型参数调优：上述每个分类算法，至少选择2个超参数，设置不同值，进行模型训练和评估，最终获取每个分类算法的最佳模型；（5分）
        //todo 逻辑回归
        lessoRa(trainData, testData,10,100);
        lessoRa(trainData, testData,6,77);
        lessoRa(trainData, testData,31,127);
        lessoRa(trainData, testData,5,17);  //todo 最佳模型
        lessoRa(trainData, testData,13,123);


        //todo 决策树分类
        treeRa(trainData, testData,10,100);
        treeRa(trainData, testData,6,77);
        treeRa(trainData, testData,31,127);  //todo 最佳模型
        treeRa(trainData, testData,5,17);
        treeRa(trainData, testData,13,123);

        //、最佳模型预测：对测试数据集进行特征工程，使用构建最佳模型分别预测，控制台显示结果；（5


    }

    //todo KNN近邻分类
    private static void lessoRa(BatchOperator<?> trainData, BatchOperator<?> testData,int i,int v) throws Exception {
        //、构建分类模型：选择Alink库中分类算法（逻辑回归、决策树分类、KNN近邻分类）任选二种，合理设置超参数值，使用trainData数据集训练构建模型；（5分）
        BatchOperator <?> trainOp = new KnnTrainBatchOp()
                .setFeatureCols("x1", "x2", "x3", "x4", "x5", "x6", "x7")
                .setLabelCol("y")
                .setDistanceType("EUCLIDEAN")
                .linkFrom(trainData);
        BatchOperator <?> predictOp = new KnnPredictBatchOp()
                .setNumThreads(i)
                .setK(v)
                .setPredictionDetailCol("pred_detail")
                .setPredictionCol("pred")
                .linkFrom(trainOp,testData);
        predictOp.print(3);

        //（7）算法模型保存：获取每个算法算法的最佳模型，并保存模型到本地文件系统；（5分）
//        CsvSinkBatchOp csvSink = new CsvSinkBatchOp()
//                .setFilePath("data/lessoRa.txt");
//
//        trainOp.link(csvSink);
//        BatchOperator.execute();


        //（4）、模型指标评估：在测试集testData上对前面构建分类模型，分别进行预测评估，计算准确率Accuracy，并比较之间差异；（5分）
        MultiClassMetrics metrics1 = new EvalMultiClassBatchOp()
                .setLabelCol("y")
                .setPredictionDetailCol("pred_detail")
                .linkFrom(predictOp)
                .collectMetrics();
        System.out.println("Prefix0 accuracy:" + metrics1.getAccuracy());
        System.out.println("Macro Precision:" + metrics1.getMacroPrecision());
        System.out.println("Micro Recall:" + metrics1.getMicroRecall());
        System.out.println("Weighted Sensitivity:" + metrics1.getWeightedSensitivity());


    }

    //todo 决策树分类
    private static void treeRa(BatchOperator<?> trainData, BatchOperator<?> testData,int i,int v) throws Exception {
        //（3）、构建分类模型：选择Alink库中分类算法（逻辑回归、决策树分类、KNN近邻分类）任选二种，合理设置超参数值，使用trainData数据集训练构建模型；（5分）
        BatchOperator <?> trainOp = new CartTrainBatchOp()
                .setLabelCol("y")
                .setFeatureCols("x1", "x2", "x3", "x4", "x5", "x6", "x7")
                .setMaxDepth(i)
                .setMaxLeaves(v)
                .linkFrom(trainData);
        BatchOperator <?> predictBatchOp = new CartPredictBatchOp()
                .setPredictionDetailCol("pred_detail")
                .setPredictionCol("pred");
        BatchOperator<?> o1 = predictBatchOp.linkFrom(trainOp, testData);

        //（7）算法模型保存：获取每个算法算法的最佳模型，并保存模型到本地文件系统；（5分）
//        CsvSinkBatchOp csvSink = new CsvSinkBatchOp()
//                .setFilePath("data/treeRa.txt");
//
//        trainOp.link(csvSink);
//        BatchOperator.execute();


        //（4）、模型指标评估：在测试集testData上对前面构建分类模型，分别进行预测评估，计算准确率Accuracy，并比较之间差异；（5分）
        MultiClassMetrics metrics1 = new EvalMultiClassBatchOp()
                .setLabelCol("y")
                .setPredictionDetailCol("pred_detail")
                .linkFrom(o1)
                .collectMetrics();
        System.out.println("Prefix0 accuracy:" + metrics1.getAccuracy());
        System.out.println("Macro Precision:" + metrics1.getMacroPrecision());
        System.out.println("Micro Recall:" + metrics1.getMicroRecall());
        System.out.println("Weighted Sensitivity:" + metrics1.getWeightedSensitivity());
    }
}
