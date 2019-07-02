using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Barracuda;

public class SpriteGenBrain : MonoBehaviour {

    public string pathToModel;
    public TextAsset graphModel;
    public int inputDimension = 100;


    // Start is called before the first frame update
    void Start() {
        Evaluate(new float[100]);
    }

    // Update is called once per frame
    void Update() {
        
    }

    void Evaluate(float[] inputVector) {
        for (int i = 0; i < 100; i++) {
            print(inputVector[i]);
        }
        var model = ModelLoader.LoadFromStreamingAssets("brain.nn", false);
        foreach (var layer in model.inputs) {
            Debug.Log(layer.ToString() + " does ");
            foreach (var m in layer.shape) {
                print(m);
            }
        }

        var worker = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputePrecompiled, model, true);
        var input = new Tensor(1, 1, 1, 100, inputVector);
        worker.Execute(input);
        var O = worker.Peek();
        print(O);


        /*    if (inputVector.Length != inputDimension) {
                Debug.LogError("Input dimension " + inputVector.Length + " is wrong. Should be " + inputDimension);
                return;
            }

            TFSession session = new TFSession(modelGraph);
            TFSession.Runner runner = session.GetRunner();

            TFTensor inputTensor = inputVector;
            runner.AddInput(modelGraph["input_x"][0], inputTensor);
            runner.Fetch(modelGraph["output_node"][0]);

            float[,,,] output = runner.Run()[0].GetValue() as float[,,,];

            session.Dispose();
            // Uncomment if issues
            //modelGraph.Dispose();
            */
    }
}

