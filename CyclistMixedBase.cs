using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace CyclingTime3
{
    class CyclistMixedBase{
        protected InferenceEngine inferenceEngine;
        protected int numComponents;
        protected VariableArray<Gaussian> averageTimePriors;
        protected VariableArray<double> averageTime;
        protected VariableArray<Gamma> trafficNoisePriors;
        protected VariableArray<double> trafficNoise;

        //still 2c 
        protected Variable<Dirichlet> mixingPrior;
        protected Variable<Vector> mixingCoefficients;

        public virtual void CreateModel(){
            inferenceEngine = new InferenceEngine(new VariationalMessagePassing());
            numComponents = 2;
            Range componentRange = new Range(numComponents);
            averageTimePriors = Variable.Array<Gaussian>(componentRange);
            trafficNoisePriors = Variable.Array<Gamma>(componentRange);
            averageTime = Variable.Array<double>(componentRange);
            trafficNoise = Variable.Array<double>(componentRange);

            using (Variable.ForEach(componentRange))
            {
                averageTime[componentRange] = Variable.Random<double, Gaussian>(averageTimePriors[componentRange]);
                trafficNoise[componentRange] = Variable.Random<double, Gamma>(trafficNoisePriors[componentRange]);
            }

            mixingPrior = Variable.New<Dirichlet>();
            mixingCoefficients = Variable<Vector>.Random(mixingPrior);
            mixingCoefficients.SetValueRange(componentRange);


        }

        public void SetModelData(ModelDataMixed modelData)
        {
            averageTimePriors.ObservedValue = modelData.averageTimeDist;
            trafficNoisePriors.ObservedValue = modelData.trafficNoiseDist;
            mixingPrior.ObservedValue = modelData.mixingDist;
        }

        public struct ModelDataMixed
        {
            public Gaussian[] averageTimeDist;
            public Gamma[] trafficNoiseDist;
            public Dirichlet mixingDist;
        }

    }
}
