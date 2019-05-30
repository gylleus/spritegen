using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class SpriteGenerator : Agent {

    public int latentVectorLen = 100;

    public override void AgentAction(float[] vectorAction, string textAction) {
        // Gör en sprite 32x32x3
    }

    public override void CollectObservations() {
        for (int i = 0; i < latentVectorLen; i++) {
            AddVectorObs(SampleNormalDist(0, 1));
        }
    }

    float SampleNormalDist(float mean, float stdev) {
        float u1 = Random.Range(0f, 1f);
        float u2 = Random.Range(0f, 1f);
        float z0 = Mathf.Sqrt(-2 * Mathf.Log(u1)) * Mathf.Sin(2 * Mathf.PI * u2);
        return mean + stdev * z0;
    }

}
