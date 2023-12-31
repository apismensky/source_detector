/*
 * Copyright (C) from 2022 The Play Framework Contributors <https://github.com/playframework>, 2011-2021 Lightbend Inc. <https://www.lightbend.com>
 */

package play.api.mvc

import play.api.http.HttpConfiguration
import play.api.mvc.request.DefaultRequestFactory
import play.core.server.netty.NettyHelpers

object MvcHelpers {
  def requestHeaderFromHeaders(headerList: List[(String, String)]): RequestHeader = {
    val channel          = NettyHelpers.nettyChannel(remoteAddress = NettyHelpers.localhost, ssl = false)
    val nettyRequest     = NettyHelpers.nettyRequest(headers = headerList)
    val convertedRequest = NettyHelpers.conversion.convertRequest(channel, nettyRequest).get
    val defaultRequest   = new DefaultRequestFactory(HttpConfiguration()).copyRequestHeader(convertedRequest)
    defaultRequest
  }
}
