using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace CyclingTime3
{
    class CyclistMixedPrediction:CyclistMixedBase
    {
        private Gaussian tomorrowsTimeDist;
        private Variable<double> tomorrowsTime;

        public override void CreateModel()
        {
            base.CreateModel();
            Variable<int> componentIndex = Variable.Discrete(mixingCoefficients);
            tomorrowsTime = Variable.New<double>();

            using (Variable.Switch(componentIndex))
            {
                tomorrowsTime.SetTo(Variable.GaussianFromMeanAndPrecision
                    (averageTime[componentIndex], trafficNoise[componentIndex]));
            }
        }
        public Gaussian InferTomorrowsTime()
        {
            tomorrowsTimeDist = inferenceEngine.Infer<Gaussian>(tomorrowsTime);
            return tomorrowsTimeDist;
        }
    }
}
