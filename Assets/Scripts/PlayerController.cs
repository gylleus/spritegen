using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour {

    public float movementSpeed;
    // Input axis is between -1 to 1 and changes continuously
    // This variable clamps the value to prevent character from moving too slow
    public float minInputAxis;

    Animator anim;

    // Start is called before the first frame update
    void Start() {
        anim = GetComponent<Animator>();
    }

    // Update is called once per frame
    void FixedUpdate() {
        float horizontalInput = Input.GetAxis("Horizontal");
        float verticalInput = Input.GetAxis("Vertical");

        if (horizontalInput > 0) {
            //        horizontalInput = horizontalInput / Mathf.Abs(horizontalInput) * Mathf.Sqrt(Mathf.Abs(horizontalInput));
            horizontalInput = Mathf.Max(horizontalInput, minInputAxis);
        } else if (horizontalInput < 0) {
            horizontalInput = Mathf.Min(horizontalInput, -minInputAxis);
        }

        if (verticalInput > 0) {
            //       verticalInput = verticalInput / Mathf.Abs(verticalInput) * Mathf.Sqrt(Mathf.Abs(verticalInput));
            verticalInput = Mathf.Max(verticalInput, minInputAxis);
        } else if (verticalInput < 0) {
            verticalInput = Mathf.Min(verticalInput, -minInputAxis);
        }

        Vector3 movementDirection = new Vector3(horizontalInput, verticalInput);

        if (movementDirection.magnitude != 0) {
            transform.Translate(movementDirection * movementSpeed);
            anim.SetBool("isRunning", true);
        } else {
            anim.SetBool("isRunning", false);
        }

    }
}
