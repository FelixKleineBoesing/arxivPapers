import React, { Component } from 'react';
import './SigmaGraph.scss';
import Sigma from "sigma";
import UndirectedGraph from 'graphology';
import circlepack from 'graphology-layout/circlepack';


class SigmaGraph extends Component {
  render() {
    return (  
    <div className="SigmaGraph" data-testid="SigmaGraph">
        <div id="sigma-container"></div>
      </div>
    );
  }

  componentDidMount() {
    this.getGraph()
  }

  getGraph() {
    const data = {"attributes": {"name": "test"}, 
    "nodes": [
      {"key": "A", "attributes": {"label": "A", "size": 10, "x": 1, "y": 2}}, 
      {"key": "B", "attributes": {"x": 3, "y": 2, "label": "B", "size": 20}}, 
      {"key": "C", "attributes": {"x": 2, "y": 3, "label": "C", "size": 15}}], 
      edges: [
      {
        key: 'A->B',
        source: 'A',
        target: 'B',
        attributes: {}
      }, 
      {
        key: 'A->C',
        source: 'A',
        target: 'C',
        attributes: {}
      }
    ]}
    const graph = UndirectedGraph.from(data);
    const container = document.getElementById("sigma-container") as HTMLElement;

    //circlepack.assign(graph);

    // Instanciate sigma:
    const renderer = new Sigma(graph, container);
    const camera = renderer.getCamera();
    
}

};


export default SigmaGraph;
