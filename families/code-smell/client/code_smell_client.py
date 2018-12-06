# Copyright 2016 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import os
import time
import random
import base64
import hashlib
import json
import requests
import yaml
import toml #pylint: disable=import-error

from pprint import pprint
from base64 import b64encode
from sawtooth_signing import ParseError #pylint: disable=import-error
from sawtooth_signing import CryptoFactory #pylint: disable=import-error
from sawtooth_signing import create_context #pylint: disable=import-error
from sawtooth_signing.secp256k1 import Secp256k1PrivateKey #pylint: disable=import-error

from sawtooth_sdk.protobuf.batch_pb2 import Batch #pylint: disable=import-error
from sawtooth_sdk.protobuf.batch_pb2 import BatchList #pylint: disable=import-error
from sawtooth_sdk.protobuf.batch_pb2 import BatchHeader #pylint: disable=import-error
from sawtooth_sdk.protobuf.transaction_pb2 import Transaction #pylint: disable=import-error
from sawtooth_sdk.protobuf.transaction_pb2 import TransactionHeader #pylint: disable=import-error

from client.code_smell_exceptions import CodeSmellException

def _sha512(data):
    """
    return hash of data
    Args:
        data (object), object to get hash
    """
    return hashlib.sha512(data).hexdigest()

class CodeSmellClient:
    """
    construct and send code smell transaction.
    """
    def __init__(self, base_url, work_path, keyfile=None):
        self._base_url = base_url
        self._work_path = work_path

        if keyfile is None:
            self._signer = None
            return

        try:
            with open(keyfile) as fileptr:
                private_key_str = fileptr.read().strip()
        except OSError as err:
            raise CodeSmellException('Failed to read private key {}: {}'.format(keyfile, str(err)))

        try:
            private_key = Secp256k1PrivateKey.from_hex(private_key_str)
        except ParseError as error:
            raise CodeSmellException('Unable to load private key: {}'.format(str(error)))

        self._signer = CryptoFactory(create_context('secp256k1')).new_signer(private_key)

    def default(self):
        """
        load a defautl code smell configuration
        """

        #identify code_smell family configuration file
        conf_file = self._work_path + '/etc/.suse'
        response = ""

        if os.path.isfile(conf_file):
            try:
                with open(conf_file) as config:
                    raw_config = config.read()
            except IOError as error:
                raise CodeSmellException("Unable to load code smell family configuration file: {}"
                                         .format(error))

            #load toml config into a dict
            parsed_toml_config = toml.loads(raw_config)

            #get default code smells
            code_smells_config = parsed_toml_config['code_smells']
            #code_smells_config = parsed_toml_config

            """traverse dict and process each code smell
                nested for loop to procces level two dict."""
            for code_smells in code_smells_config.values():
                for name, metric in code_smells.items():
                    #send trasaction
                    response = self._send_code_smell_txn(
                        txn_type='code_smell',
                        txn_id=name,
                        data=str(metric[0]), ## TODO: add weigth value
                        state='create')

            code_smells_config = parsed_toml_config['vote_setting']

            """traverse dict and process each code smell
                nested for loop to procces level two dict."""
            for name, metric in code_smells_config.items():
                #send transaction
                response = self._send_code_smell_txn(
                    txn_type='code_smell',
                    txn_id=name,
                    data=str(metric[0]),
                    state='create')
        else:
            raise CodeSmellException("Configuration File {} does not exists".format(conf_file))

        return response

    def list(self, txn_type=None, active_flag=None):
        """
        list all transactions.

        Args:
            type (str), asset that we want to list (code smells, proposals, votes)
        """
        #pull all transactions of code smell family
        result = self._send_request("transactions")

        transactions = {}
        try:
            encoded_entries = yaml.safe_load(result)["data"]
            if txn_type is None:
                for entry in encoded_entries:
                    transactions[entry["header_signature"]] = base64.b64decode(entry["payload"])

            else:
                for entry in encoded_entries:
                    transaction_type = base64.b64decode(entry["payload"]).decode().split(',')[0]
                    if transaction_type == txn_type:
                        transactions[entry["header_signature"]] = base64.b64decode(entry["payload"])

            if txn_type == 'proposal' and active_flag == 1:
                return sorted(transactions)
            else:
                return transactions
        except BaseException:
            return None

    def show(self, address):
        """
        list a specific transaction, based on its address

        Args:
            address (str), transaction's address
        """
        result = self._send_request("transactions/{}".format(address))

        transactions = {}
        try:
            encoded_entries = yaml.safe_load(result)["data"]
            transactions["payload"] = base64.b64decode(encoded_entries["payload"])
            transactions["header_signature"] = encoded_entries["header_signature"]
            return transactions
        except BaseException:
            return None

    def propose(self, code_smells):
        """
        propose new metrics for the code smell families
        the function assumes that all code smells have been updated even if they don't

        Args:
            code_smells (dict), dictionary of code smells and metrics
        """
        #get code smell family address prefix
        code_smell_prefix = self._get_prefix()

        #check for an active proposal, transactions are return in sequential order.
        proposal_result = self._send_request("state?address={}".format(code_smell_prefix))
        encoded_entries = yaml.safe_load(proposal_result)["data"]
        for entry in encoded_entries:
            #look for the first proposal transactiosn
            if base64.b64decode(entry["data"]).decode().split(',')[0] == "proposal":
                last_proposal = base64.b64decode(entry["data"]).decode().split(',')
                break
        try:
            if last_proposal[3] == "active":
                return "Invalid Operation, another proposal is Active"
        except BaseException:
            pass

        localtime = time.localtime(time.time())
        transac_time = str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday)
        propose_date = str(transac_time)

        response = self._send_code_smell_txn(
            txn_id=_sha512(str(code_smells).encode('utf-8'))[0:6],
            txn_type='proposal',
            data=str(code_smells).replace(",", ";"),
            state='active',
            date=propose_date)

        return response

    def update_proposal(self, proposal_id, state):
        """
        update proposal state

        Args:
            proposal_id (str), proposal ID
            state (Str), new proposal ID
        """
        proposal = self.show(proposal_id)
        self._update_proposal(proposal, state)

    def _update_proposal(self, proposal, state):
        """
        update proposal, update state.

        Args:
            proposal (dict), proposal data
            sate (str), new proposal's state
        """
        localtime = time.localtime(time.time())
        transac_time = str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday)
        propose_date = str(transac_time)

        response = self._send_code_smell_txn(
            txn_id=proposal[1],
            txn_type='proposal',
            data=proposal[2],
            state=state,
            date=propose_date)

        work_path = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        conf_file = work_path + '/etc/.suse'

        if os.path.isfile(conf_file):
            try:
                with open(conf_file) as config:
                    raw_config = config.read()
            except IOError as error:
                raise CodeSmellException("Unable to load code smell family configuration file: {}"
                                         .format(error))
            #load toml config into a dict
            toml_config = toml.loads(raw_config)
        self._send_git_request(toml_config)

    def _send_git_request(self, toml_config):
        """
        send new code smell configuration to github

        Args:
            toml_config (dict): code smells to send
        """
        wrapper_json = {}
        wrapper_json["sender"] = "Sawtooth"
        wrapper_json["repo"] = "157484644" ## TODO: update with dynamic repo
        wrapper_json["suse_file"] = toml_config
        data = json.dumps(wrapper_json)
        requests.post('http://129.108.7.2:3000', data=data)

    def update_config(self, proposal):
        """
        update code smell configuration metrics, after the proposal is accepted
        the configuration file needs to be updated.

        Args:
            toml_config (dict), current configuration
            proposal (str), proposal that contains new configuration
        """
        #get proposal payload
        proposal_payload = yaml.safe_load(proposal[2].replace(";", ","))

        work_path = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        #identify code_smell family configuration file
        conf_file = work_path + '/etc/.suse'

        if os.path.isfile(conf_file):
            try:
                with open(conf_file) as config:
                    raw_config = config.read()
            except IOError as error:
                raise CodeSmellException("Unable to load code smell family configuration file: {}"
                                         .format(error))
            #load toml config into a dict
            toml_config = toml.loads(raw_config)

        """
        start by traversing the proposal,
        get the code smell and the metric
        """
        for proposal_key, proposal_metric in proposal_payload.items():
            tmp_type = ""
            """
            we don't know where on the toml file is the code smell,
            traverse the toml dictionary looking for the same code smell.
            """
            for code_type in toml_config["code_smells"]:
                """
                once you found the code smell, break the loop and return
                a pseudo location
                """
                if proposal_key in toml_config["code_smells"][code_type].keys():
                    tmp_type = code_type
                    break
            #update configuration
            toml_config["code_smells"][tmp_type][proposal_key][0] = int(proposal_metric)

        #save new configuration
        try:
            with open(conf_file, 'w+') as config:
                toml.dump(toml_config, config)
            #self._send_git_request(toml_config)
        except IOError as error:
            raise CodeSmellException("Unable to open configuration file {}".format(error))

    def check_votes(self, proposal_id):
        """
        review the votes of a proposal

        Args:
            proposal_id (str), proposal id
        """
        result = self._send_request("transactions/{}".format(proposal_id))
        encoded_result = yaml.safe_load(result)["data"]
        proposal = base64.b64decode(encoded_result["payload"]).decode().split(',')
        proposal_id = proposal[1]
        transactions = self.list(txn_type='vote')
        votes = []
        for vote in transactions:
            #for all votes of proposal
            if transactions[vote].decode().split(',')[2] == proposal_id:
                #get vote and count, only accepted votes
                #if transactions[vote].decode().split(',')[3] == '1':
                votes.append(int(transactions[vote].decode().split(',')[3]))
        return votes

        ##################
        #NO NEED THE SERVER WILL HANDLE THIS
        #get treshold
        #identify code_smell family configuration file
        # conf_file = self._work_path + '/etc/.suse'
        #
        # if flag is not None:
        #     if os.path.isfile(conf_file):
        #         try:
        #             with open(conf_file) as config:
        #                 raw_config = config.read()
        #                 config.close()
        #         except IOError as error:
        #             raise CodeSmellException(
        #                 "Unable to load code smell family configuration file {}".format(error))
        #
        #         #load toml config into a dict
        #         parsed_toml_config = toml.loads(raw_config)
        #
        #         #get treshold
        #         code_smells_config = parsed_toml_config['vote_setting']
        #
        #         vote_treshold = int(code_smells_config['approval_treshold'])
        #
        #         if total_votes >= vote_treshold:
        #             self._update_config(parsed_toml_config, proposal)
        #             #you need this, you commented out to test the github stuff
        #             #self._update_proposal(proposal, "accepted")
        # else:
        #     return "Total votes (accepted): " + str(total_votes)

    def vote(self, proposal_id, vote):
        """
        vote to accept or reject a proposal

        Args:
            proposal_id (str), id of proposal
            vote (int), value of vote 1=accept, 0=reject
        """

        #verify active proposal
        result = self._send_request("transactions/{}".format(proposal_id))
        encoded_result = yaml.safe_load(result)["data"]
        proposal = base64.b64decode(encoded_result["payload"]).decode().split(',')
        if proposal[3] != 'active':
            return "Proposal not active"

        #verify double voting
        # proposal_id = proposal[1]
        # result = self._send_request("transactions")
        # encoded_entries = yaml.safe_load(result)["data"]
        # for entry in encoded_entries:
        #     transaction_type = base64.b64decode(entry["payload"]).decode().split(',')[0]
        #     if transaction_type == 'vote':
        #         if entry['header']['signer_public_key'] == self._signer.get_public_key().as_hex():
        #             return ("User already submitted a vote")

        #active proposal, record vote
        response = self._send_code_smell_txn(
            txn_id=str(random.randrange(1, 99999)),
            txn_type='vote',
            data=proposal[1],
            state=str(vote))

        # conf_file = '/home/mrwayne/Desktop/Susereum/Sawtooth/etc/.suse'
        #
        # if os.path.isfile(conf_file):
        #     try:
        #         with open(conf_file) as config:
        #             raw_config = config.read()
        #             config.close()
        #     except IOError as error:
        #         raise CodeSmellException(
        #             "Unable to load code smell family configuration file {}".format(error))
        #
        #     #load toml config into a dict
        # parsed_toml_config = toml.loads(raw_config)
        # self._send_git_request(parsed_toml_config)

        return response

    def send_config(self, config=None):
        """
        function to send an update configuration transaction to the chain
        after the code smell configuration is update al peers in the network
        must update the local configuration file.

        Args:
            config (dictionary), code smell configuration
        """
        #read .suse configuration file
        toml_config = self._get_config_file()
        print (toml_config)

    def _get_config_file(self):
        work_path = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

        #identify code_smell family configuration file
        conf_file = work_path + '/etc/.suse'

        if os.path.isfile(conf_file):
            try:
                with open(conf_file) as config:
                    raw_config = config.read()
            except IOError as error:
                raise CodeSmellException("Unable to load code smell family configuration file: {}"
                                         .format(error))
        #load toml config into a dict
        toml_config = toml.loads(raw_config)
        return toml_config

    def _get_status(self, batch_id, wait, auth_user=None, auth_password=None):
        try:
            result = self._send_request(
                'batch_status?id={}&wait={}'.format(batch_id, wait),
                auth_user=auth_user,
                auth_password=auth_password)
            return yaml.safe_load(result)['data'][0]['status']
        except BaseException as err:
            raise CodeSmellException(err)

    def _get_prefix(self):
        """
        get code smell family address prefix
        """
        return _sha512('code-smell'.encode('utf-8'))[0:6]

    def _get_address(self, transaction_id):
        """
        get transaction address

        Args:
            id (str): trasaction id
        """
        code_smell_prefix = self._get_prefix()
        code_smell_address = _sha512(transaction_id.encode('utf-8'))[0:64]
        return code_smell_prefix + code_smell_address

    def _send_request(self,
                      suffix,
                      data=None,
                      content_type=None,
                      auth_user=None,
                      auth_password=None):
        """
        send request to code smell processor`
        """
        if self._base_url.startswith("http://"):
            url = "{}/{}".format(self._base_url, suffix)
        else:
            url = "http://{}/{}".format(self._base_url, suffix)

        headers = {}
        if auth_user is not None:
            auth_string = "{}:{}".format(auth_user, auth_password)
            b64_string = b64encode(auth_string.encode()).decode()
            auth_header = 'Basic {}'.format(b64_string)
            headers['authorization'] = auth_header

        if content_type is not None:
            headers['Content-Type'] = content_type

        try:
            if data is not None:
                result = requests.post(url, headers=headers, data=data)
            else:
                result = requests.get(url, headers=headers)

            if result.status_code == 404:
                raise CodeSmellException("No such transaction")
            elif not result.ok:
                raise CodeSmellException("Error {}:{}".format(result.status_code, result.reason))

        except requests.ConnectionError as err:
            raise CodeSmellException('Failed to connect to {}:{}'.format(url, str(err)))

        except BaseException as err:
            raise CodeSmellException(err)

        return result.text

    def _send_code_smell_txn(self,
                             txn_type=None,
                             txn_id=None,
                             data=None,
                             state=None,
                             date=None):
        """
        serialize payload and create header transaction

        Args:
            type (str):    type of transaction
            id (str):      asset id, will depend on type of transaction
            data (object): transaction data
            state (str):   all transactions must have a state
            wait (int):    delay to process transactions
        """
        #serialization is just a delimited utf-8 encoded strings
        if txn_type == 'proposal':
            payload = ",".join([txn_type, txn_id, data, state, str(date)]).encode()
        else:
            payload = ",".join([txn_type, txn_id, data, state]).encode()

        pprint("payload: {}".format(payload))######################################## pprint

        #construct the address
        address = self._get_address(txn_id)

        #construct header`
        header = TransactionHeader(
            signer_public_key=self._signer.get_public_key().as_hex(),
            family_name="code-smell",
            family_version="0.1",
            inputs=[address],
            outputs=[address],
            dependencies=[],
            payload_sha512=_sha512(payload),
            batcher_public_key=self._signer.get_public_key().as_hex(),
            nonce=hex(random.randint(0, 2**64))
        ).SerializeToString()

        signature = self._signer.sign(header)

        #create transaction
        transaction = Transaction(
            header=header,
            payload=payload,
            header_signature=signature
        )

        #create batch list, suserum policy: one transaction per batch
        batch_list = self._create_batch_list([transaction])

        return self._send_request(
            "batches",
            batch_list.SerializeToString(),
            'application/octet-stream')

    def _create_batch_list(self, transactions):
        """
        Create the list of batches that the client will send to the REST API

        Args:
            transactions (transaction): transaction(s) included in the batch

        Returns:
            BatchList: a list of batches to send to the REST API
        """
        transaction_signatures = [t.header_signature for t in transactions]

        header = BatchHeader(
            signer_public_key=self._signer.get_public_key().as_hex(),
            transaction_ids=transaction_signatures
        ).SerializeToString()

        signature = self._signer.sign(header)

        batch = Batch(
            header=header,
            transactions=transactions,
            header_signature=signature)

        return BatchList(batches=[batch])
