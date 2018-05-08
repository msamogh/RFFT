import React from 'react';
import constants from '../../constants';
import './Home.css';
class ExperimentCard extends React.Component {
  getDomainName = (domainId) => {
    switch(domainId) {
      case 0:
        return 'Text';
      case 1:
        return 'Image';
      case 2:
        return 'Tabular';
      default:
        return 'Other';
    }
  }

  render() {
    const {
      name, description, domain, started,
    } = this.props.experiment;
    return (
      <div className="Home-ExperimentCard">
        <div className="Home-ExperimentCard-info">
          <h1 className="Home-ExperimentCard-name">{name}</h1>
          <p className="Home-ExperimentCard-desc">{description}</p>
          <h2 className="Home-ExperimentCard-domain">{this.getDomainName(domain)}</h2>
        </div>
        <button className="Home-ExperimentCard-button" onClick={this.props.goToWorkspace}>{started ? 'Resume experiment' : 'Start experiment'}</button>
      </div>
    );
  }
}

class ExperimentList extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      all_experiments: [],
    };
  }

  componentDidMount = () => {
    const API = `${constants.API}/all_experiments`;
    fetch(API)
      .then(response => response.json())
      .then(data => this.setState({ all_experiments: data.all_experiments }));
  }

  goToWorkspace = (experiment) => () => {
    this.props.goToWorkspace(experiment);
  }

  render() {
    return this.state.all_experiments.map(experiment => (
      <ExperimentCard key={experiment.id} experiment={experiment} goToWorkspace={this.goToWorkspace(experiment)}/>
    ))
  }
}

class Home extends React.Component {
  render() {
    return (
      <div className="Home">
        <ExperimentList goToWorkspace={this.props.goToWorkspace}/>
      </div>
    );
  }
}

export default Home;
