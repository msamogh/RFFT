import React from 'react';
import './Home.css';


const API = 'http://whitebox-rfft.herokuapp.com/api/v1';


class ExperimentCard extends React.Component {
  getDomainName = (domainId) => {
    switch(domainId) {
      case 0:
        return 'Text';
      case 1:
        return 'Image';
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

class Home extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      all_experiments: [],
    };
  }

  componentDidMount = () => {
    // TODO get running experiments
    fetch(API + '/all_experiments')
      .then(response => response.json())
      .then(data => this.setState({ all_experiments: data.all_experiments }));

    // this.setState({
    //   all_experiments: [
    //     {
    //       name: 'Decoy MNIST',
    //       description: 'Handwritten digits',
    //       domain: 1,
    //       started: true,
    //     },
    //     {
    //       name: 'Newsgroup-20',
    //       description: 'Emails',
    //       domain: 0,
    //       started: false,
    //     },
    //   ],
    // });
  }

  goToWorkspace = (experiment) => () => {this.props.goToWorkspace(experiment);}

  renderBody = () => this.state.all_experiments.map(experiment => (<ExperimentCard experiment={experiment} goToWorkspace={this.goToWorkspace(experiment)}/>))

  render() {
    return (
      <div className="Home">
        {this.renderBody()}
      </div>
    );
  }
}

export default Home;
