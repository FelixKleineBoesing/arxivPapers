import React from 'react';
import './App.css';
import Navbar from './components/Navbar/Navbar';
import Charts from './components/Charts/Charts';

function App() {
  return (
    <div className="App">
      <Navbar />
      <body>
        <Charts />
      </body>
    </div>
  );
}

export default App;
