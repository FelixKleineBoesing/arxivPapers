import './Navbar.scss';
import Navbar from 'react-bootstrap/Navbar'
import React, { Component } from "react";
import { Container, Nav, NavDropdown } from 'react-bootstrap';

class NavBar extends Component {
  render() {
    return (
      <Navbar bg="light" expand="lg">
        <Container>
          <Navbar.Brand href="home">Felix Kleine Bösing</Navbar.Brand>
          <Navbar.Brand href="arxiv">Arxiv Computer Science Paper</Navbar.Brand>
         </Container>
      </Navbar>
    );
  }
}

export default NavBar;