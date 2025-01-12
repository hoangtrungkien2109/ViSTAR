# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import streaming_pb2 as streaming__pb2

GRPC_GENERATED_VERSION = '1.64.1'
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = '1.65.0'
SCHEDULED_RELEASE_DATE = 'June 25, 2024'
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in streaming_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class StreamingStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.PushText = channel.unary_unary(
                '/streaming_services.Streaming/PushText',
                request_serializer=streaming__pb2.PushTextRequest.SerializeToString,
                response_deserializer=streaming__pb2.PushTextResponse.FromString,
                _registered_method=True)
        self.PopText = channel.unary_unary(
                '/streaming_services.Streaming/PopText',
                request_serializer=streaming__pb2.PopTextRequest.SerializeToString,
                response_deserializer=streaming__pb2.PopTextResponse.FromString,
                _registered_method=True)
        self.PushFrame = channel.unary_unary(
                '/streaming_services.Streaming/PushFrame',
                request_serializer=streaming__pb2.PushFrameRequest.SerializeToString,
                response_deserializer=streaming__pb2.PushFrameResponse.FromString,
                _registered_method=True)
        self.PopFrame = channel.unary_unary(
                '/streaming_services.Streaming/PopFrame',
                request_serializer=streaming__pb2.PopFrameRequest.SerializeToString,
                response_deserializer=streaming__pb2.PopFrameResponse.FromString,
                _registered_method=True)
        self.PushImage = channel.unary_unary(
                '/streaming_services.Streaming/PushImage',
                request_serializer=streaming__pb2.PushImageRequest.SerializeToString,
                response_deserializer=streaming__pb2.PushImageResponse.FromString,
                _registered_method=True)
        self.PopImage = channel.unary_unary(
                '/streaming_services.Streaming/PopImage',
                request_serializer=streaming__pb2.PopImageRequest.SerializeToString,
                response_deserializer=streaming__pb2.PopImageResponse.FromString,
                _registered_method=True)


class StreamingServicer(object):
    """Missing associated documentation comment in .proto file."""

    def PushText(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PopText(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PushFrame(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PopFrame(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PushImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PopImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_StreamingServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'PushText': grpc.unary_unary_rpc_method_handler(
                    servicer.PushText,
                    request_deserializer=streaming__pb2.PushTextRequest.FromString,
                    response_serializer=streaming__pb2.PushTextResponse.SerializeToString,
            ),
            'PopText': grpc.unary_unary_rpc_method_handler(
                    servicer.PopText,
                    request_deserializer=streaming__pb2.PopTextRequest.FromString,
                    response_serializer=streaming__pb2.PopTextResponse.SerializeToString,
            ),
            'PushFrame': grpc.unary_unary_rpc_method_handler(
                    servicer.PushFrame,
                    request_deserializer=streaming__pb2.PushFrameRequest.FromString,
                    response_serializer=streaming__pb2.PushFrameResponse.SerializeToString,
            ),
            'PopFrame': grpc.unary_unary_rpc_method_handler(
                    servicer.PopFrame,
                    request_deserializer=streaming__pb2.PopFrameRequest.FromString,
                    response_serializer=streaming__pb2.PopFrameResponse.SerializeToString,
            ),
            'PushImage': grpc.unary_unary_rpc_method_handler(
                    servicer.PushImage,
                    request_deserializer=streaming__pb2.PushImageRequest.FromString,
                    response_serializer=streaming__pb2.PushImageResponse.SerializeToString,
            ),
            'PopImage': grpc.unary_unary_rpc_method_handler(
                    servicer.PopImage,
                    request_deserializer=streaming__pb2.PopImageRequest.FromString,
                    response_serializer=streaming__pb2.PopImageResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'streaming_services.Streaming', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('streaming_services.Streaming', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class Streaming(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def PushText(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/streaming_services.Streaming/PushText',
            streaming__pb2.PushTextRequest.SerializeToString,
            streaming__pb2.PushTextResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def PopText(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/streaming_services.Streaming/PopText',
            streaming__pb2.PopTextRequest.SerializeToString,
            streaming__pb2.PopTextResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def PushFrame(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/streaming_services.Streaming/PushFrame',
            streaming__pb2.PushFrameRequest.SerializeToString,
            streaming__pb2.PushFrameResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def PopFrame(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/streaming_services.Streaming/PopFrame',
            streaming__pb2.PopFrameRequest.SerializeToString,
            streaming__pb2.PopFrameResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def PushImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/streaming_services.Streaming/PushImage',
            streaming__pb2.PushImageRequest.SerializeToString,
            streaming__pb2.PushImageResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def PopImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/streaming_services.Streaming/PopImage',
            streaming__pb2.PopImageRequest.SerializeToString,
            streaming__pb2.PopImageResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
