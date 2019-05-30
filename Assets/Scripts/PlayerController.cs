using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour {

    public float movementSpeed;

    Animator anim;

    // Start is called before the first frame update
    void Start() {
        anim = GetComponent<Animator>();
    }

    // Update is called once per frame
    void FixedUpdate() {
        float horizontalInput = Input.GetAxis("Horizontal");
        float verticalInput = Input.GetAxis("Vertical");

        Vector3 movementDirection = new Vector3(horizontalInput, verticalInput);

        if (movementDirection.magnitude != 0) {
            transform.Translate(movementDirection * movementSpeed);
            anim.SetBool("isRunning", true);
        } else {
            anim.SetBool("isRunning", false);
        }

    }
}
