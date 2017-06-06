using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using static CyclingTime3.CyclistMixedBase;

namespace CyclingTime3 {
    class Program3 {
        static void Main(string[] args) {
            ModelDataMixed initpriors;

            double[] trainingData = new double[] { 13, 17, 16, 12, 13, 12, 14, 18, 16, 16, 27, 32 };

            initpriors.averageTimeDist = new Gaussian[] { new Gaussian(15.0, 100.0), new Gaussian(30.0, 100.00) };
            initpriors.trafficNoiseDist = new Gamma[] { new Gamma(2.0, 0.5), new Gamma(2.0, 0.5) };
            initpriors.mixingDist = new Dirichlet(1, 1);

            CyclingMixedTraining cyclingMixedTraining = new CyclingMixedTraining();
            cyclingMixedTraining.CreateModel();
            cyclingMixedTraining.SetModelData(initpriors);
            ModelDataMixed posteriors = cyclingMixedTraining.InferModelData(trainingData);

            CyclistMixedPrediction cyclistMixedPrediction = new CyclistMixedPrediction();
            cyclistMixedPrediction.CreateModel();
            cyclistMixedPrediction.SetModelData(posteriors);

            Gaussian tomorrowsTime = cyclistMixedPrediction.InferTomorrowsTime();

            Console.WriteLine("Tomorrow's predicted time is {0:f2}", tomorrowsTime.GetMean());
            Console.WriteLine("Tomorrow's predicted standard deviation is {0:f2}", Math.Sqrt(tomorrowsTime.GetVariance()));

            Console.ReadKey();

        }
    }
}
