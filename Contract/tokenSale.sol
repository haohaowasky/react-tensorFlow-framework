pragma solidity ^0.4.19;

import './ERC223.sol';

contract TokenSale {

  address public owner;
  //address public wallet;
  uint public price;
  ERC223 public token;

  function TokenSale(
    //address _wallet,
    uint256 _tokenSupply,
    string _tokenName,
    uint8 _tokenDecimals,
    string _tokenSymbol,
    uint _price
    ) {
    owner = msg.sender;
    //wallet = _wallet;
    token = new ERC223(_tokenSupply, _tokenName, _tokenDecimals, _tokenSymbol,this);
    price = _price;

    //assert(token.balanceOf(this) == token.totalSupply());
    }

  function getTokens(uint _amount) public{
      token.transfer(msg.sender,_amount);
  }
  function getBalance(address _address) public view returns (uint){
      return token.balanceOf(_address);
  }

  function sendToken(address _to, uint256 _value) public {
     token.transfer( _to, _value);
  }
}
