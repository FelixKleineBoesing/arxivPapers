import React from 'react';
import './Navbar.scss';
import

class Navbar extends React.Component {
    render() {
    return (
        <Navbar className="navbar navbar-expand-lg fixed-top is-white is-dark-text">
              <div className="navbar-brand h1 mb-0 text-large font-medium">
                Online Retail Dashboard
              </div>
              <div className="navbar-nav ml-auto">
                <div className="user-detail-section">
                  <span className="pr-2">Hi, Sean</span>
              </div>
          </div>
        </Navbar>
        );
    }
}

export default Navbar;
