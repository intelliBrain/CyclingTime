using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace CyclingTime3
{
    class CyclingMixedTraining:CyclistMixedBase
    {
        protected Variable<int> numTrips;
        protected VariableArray<double> travelTimes;
        protected VariableArray<int> componentIndices;

        public override void CreateModel()
        {
            base.CreateModel();
            numTrips = Variable.New<int>();
            Range tripRange = new Range(numTrips);
            travelTimes = Variable.Array<double>(tripRange);
            componentIndices = Variable.Array<int>(tripRange);
            using (Variable.ForEach(tripRange))
            {
                componentIndices[tripRange] = Variable.Discrete(mixingCoefficients);
                using (Variable.Switch(componentIndices[tripRange]))
                {
                    travelTimes[tripRange].SetTo(Variable.GaussianFromMeanAndPrecision
                        (averageTime[componentIndices[tripRange]], trafficNoise[componentIndices[tripRange]]));
                }
            }
        }

        public ModelDataMixed InferModelData(double[] trainingData)
        {
            ModelDataMixed posteriors;

            travelTimes.ObservedValue = trainingData;
            numTrips.ObservedValue = trainingData.Length;

            posteriors.averageTimeDist = inferenceEngine.Infer<Gaussian[]>(averageTime);
            posteriors.trafficNoiseDist = inferenceEngine.Infer<Gamma[]>(trafficNoise);
            posteriors.mixingDist = inferenceEngine.Infer<Dirichlet>(mixingCoefficients);

            return posteriors;
        }
    }
}
