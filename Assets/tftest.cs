using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using TensorFlow;

public class tftest : MonoBehaviour
{
	// third value corresponds to inputWidth
	public TextAsset graphModel;
  	// the dimenstions of 'inputTensor' correspond to the dimensions of the input for my trained graph
    public int latentVectorLen = 100;
  	private float[,] inputTensor = new float[1, 100];

    // declare a list for outputs
    float[,] recurrentTensor;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    //void Update()
    //{
        
    //}

    void Update()
    {
           // set up and run a graph session
	      using (var graph = new TFGraph())
	      {
		  	graph.Import(graphModel.bytes);
		  	var session = new TFSession(graph);

		  	var runner = session.GetRunner();

		  	// implicitally convert a C# array to a tensor
            inputTensor = GetLatentVector();
		  	TFTensor input = inputTensor;
            Debug.Log(inputTensor);  

		  	// set up input tensor and input
		  	// KEY: I am telling my session to go find the placeholder named "input_placeholder_x" and user my input TENSOR instead
		  	runner.AddInput(graph["input_placeholder_x"][0], input);

		  	// set up output tensor
		  	runner.Fetch(graph["output_node"][0]);

		  	// run model - recurrentTensor now holds the probabilities for each outcome
		  	recurrentTensor = runner.Run()[0].GetValue() as float[,];

		  	// frees up resources - very important if you are running graph > 400 or so times
			session.Dispose();
			graph.Dispose();
            }
    }

    private float[,] GetLatentVector() {
        inputTensor = new float[1, 100];
        for (int i = 0; i < latentVectorLen; i++) {
            inputTensor[0, i] = SampleNormalDist(0, 1);
        }
        return inputTensor;
    }

    float SampleNormalDist(float mean, float stdev) {
        float u1 = Random.Range(0f, 1f);
        float u2 = Random.Range(0f, 1f);
        float z0 = Mathf.Sqrt(-2 * Mathf.Log(u1)) * Mathf.Sin(2 * Mathf.PI * u2);
        return mean + stdev * z0;
    }
}
