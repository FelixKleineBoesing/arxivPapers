import React, { Component } from 'react';
import SigmaGraph from '../SigmaGraph/SigmaGraph';
import Statistics from '../Statistics/Statistics';
import './Charts.scss';


class Charts extends Component {
  render() {
    return (
      <div className="Charts" data-testid="Charts">
        <div className="firstRow">
        <SigmaGraph />
        <Statistics />
        </div>
      </div>
    ); 
  } 
}

export default Charts;
