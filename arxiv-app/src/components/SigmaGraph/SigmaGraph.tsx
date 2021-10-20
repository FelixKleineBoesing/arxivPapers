import React, { Component } from 'react';
import './SigmaGraph.scss';
import Sigma from "sigma";
import UndirectedGraph from 'graphology';
import circlepack from 'graphology-layout/circlepack';


class SigmaGraph extends Component {
  render() {
    return (  
    <div className="SigmaGraph" data-testid="SigmaGraph">
      SigmaGraph Component
        <div id="sigma-container"></div>
        <div id="controls">
          <div className="input"><label htmlFor="zoom-in">Zoom in</label><button id="zoom-in">+</button></div>
          <div className="input"><label htmlFor="zoom-out">Zoom out</label><button id="zoom-out">-</button></div>
          <div className="input"><label htmlFor="zoom-reset">Reset zoom</label><button id="zoom-reset">⊙</button></div>
          <div className="input">
            <label htmlFor="labels-threshold">Labels threshold</label>
            <input id="labels-threshold" type="range" min="0" max="15" step="0.5" />
          </div>
        </div>
      </div>
    );
  }

  componentDidMount() {
    this.getGraph()
  }

  getGraph() {
    const data = {"attributes": {"name": "test"}, "nodes": [{"key": "A", "x": 1, "y": 2}, {"key": "B", "x": 1, "y": 2}], edges: [
      {
        key: 'A->B',
        source: 'A',
        target: 'B',
        attributes: {}
      }
    ]}
    const graph = UndirectedGraph.from(data);
    const container = document.getElementById("sigma-container") as HTMLElement;
    const zoomInBtn = document.getElementById("zoom-in") as HTMLButtonElement;
    const zoomOutBtn = document.getElementById("zoom-out") as HTMLButtonElement;
    const zoomResetBtn = document.getElementById("zoom-reset") as HTMLButtonElement;
    const labelsThresholdRange = document.getElementById("labels-threshold") as HTMLInputElement;

    circlepack.assign(graph);

    // Instanciate sigma:
    const renderer = new Sigma(graph, container);
    const camera = renderer.getCamera();

    // Bind zoom manipulation buttons
    zoomInBtn.addEventListener("click", () => {
      camera.animatedZoom({ duration: 600 });
    });
    zoomOutBtn.addEventListener("click", () => {
      camera.animatedUnzoom({ duration: 600 });
    });
    zoomResetBtn.addEventListener("click", () => {
      camera.animatedReset({ duration: 600 });
    });

    // Bind labels threshold to range input
    labelsThresholdRange.addEventListener("input", () => {
      renderer.setSetting("labelRenderedSizeThreshold", +labelsThresholdRange.value);
    });

    // Set proper range initial value:
    labelsThresholdRange.value = renderer.getSetting("labelRenderedSizeThreshold") + "";
  };
    
}




export default SigmaGraph;
