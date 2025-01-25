import React from 'react';

function Signup() {
  return (
    <div className="container">
      <h1>Signup</h1>
      <input type="text" placeholder="Username" />
      <input type="password" placeholder="Password" />
      <button>Sign Up</button>
      <p>
        Already have an account? <a href="/">Login</a>
      </p>
    </div>
  );
}

export default Signup;
