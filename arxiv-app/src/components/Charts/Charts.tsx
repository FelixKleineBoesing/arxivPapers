import React, { Component } from 'react';
import SigmaGraph from '../SigmaGraph/SigmaGraph';
import './Charts.scss';


class Charts extends Component {
  render() {
    return (
      <div className="Charts" data-testid="Charts">
        Charts Component
        <div>
        <SigmaGraph />
        </div>
      </div>
    ); 
  } 
}

export default Charts;
